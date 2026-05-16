import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchdiffeq import odeint

from config import RANDOM_SEED
from utils import append_result_csv, compute_mae_rmse_r2, ensure_dir, set_seed


RESULT_COLUMNS = [
    "protocol",
    "split_name",
    "model",
    "input_type",
    "best_params",
    "mae",
    "rmse",
    "r2",
    "num_train_bearings",
    "num_val_bearings",
    "num_test_bearings",
]


class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class FiLM(nn.Module):
    def __init__(self, condition_dim: int, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(condition_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2 * feature_dim),
        )

    def forward(self, x, condition):
        gamma, beta = self.net(condition).chunk(2, dim=-1)
        return x * (1.0 + gamma) + beta


class ConditionalEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16, condition_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.film = FiLM(condition_dim, 64)
        self.fc2 = nn.Linear(64, latent_dim)

    def forward(self, x, condition):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.film(h, condition))
        return self.fc2(h)


class PlainODEFunc(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, t, z):
        return self.net(z)


class ConditionAwareODEFunc(nn.Module):
    def __init__(self, latent_dim: int = 16, condition_dim: int = 2):
        super().__init__()
        self.condition = None
        self.fc1 = nn.Linear(latent_dim + condition_dim, 64)
        self.film1 = FiLM(condition_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.film2 = FiLM(condition_dim, 64)
        self.fc3 = nn.Linear(64, latent_dim)

    def set_condition(self, condition):
        self.condition = condition

    def forward(self, t, z):
        if self.condition is None:
            raise RuntimeError("ConditionAwareODEFunc condition was not set before integration.")
        if self.condition.shape[0] != z.shape[0]:
            raise RuntimeError("Condition batch size does not match latent batch size.")
        h = torch.tanh(self.fc1(torch.cat([z, self.condition], dim=-1)))
        h = torch.tanh(self.film1(h, self.condition))
        h = torch.tanh(self.fc2(h))
        h = torch.tanh(self.film2(h, self.condition))
        return self.fc3(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


class ConditionalDecoder(nn.Module):
    def __init__(self, latent_dim: int = 16, condition_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.film = FiLM(condition_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, z, condition):
        h = torch.relu(self.fc1(z))
        h = torch.relu(self.film(h, condition))
        return torch.sigmoid(self.fc2(h)).squeeze(-1)


class LatentODERegressor(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16, model_type: str = "latent_ode"):
        super().__init__()
        self.model_type = model_type
        if model_type == "latent_ode":
            self.encoder = Encoder(input_dim, latent_dim)
            self.ode_func = PlainODEFunc(latent_dim)
            self.decoder = Decoder(latent_dim)
        elif model_type == "condition_aware_ode":
            self.encoder = ConditionalEncoder(input_dim, latent_dim, condition_dim=2)
            self.ode_func = ConditionAwareODEFunc(latent_dim, condition_dim=2)
            self.decoder = ConditionalDecoder(latent_dim, condition_dim=2)
        else:
            raise ValueError(f"Unknown ODE model type: {model_type}")

    def forward(self, X, condition=None, tau=None):
        batch_size, k, _ = X.shape
        if self.model_type == "condition_aware_ode":
            if condition is None:
                raise RuntimeError("condition_aware_ode requires condition input.")
            z0 = self.encoder(X[:, 0, :], condition)
        else:
            z0 = self.encoder(X[:, 0, :])

        if tau is None:
            tau = torch.linspace(0.0, 1.0, k, device=X.device, dtype=X.dtype).repeat(batch_size, 1)
        else:
            tau = tau.to(device=X.device, dtype=X.dtype)

        z_traj = self._integrate(z0, tau, condition)
        z_final = z_traj[-1]
        if self.model_type == "condition_aware_ode":
            y_pred = self.decoder(z_final, condition)
        else:
            y_pred = self.decoder(z_final)
        return y_pred, z_traj

    def _integrate(self, z0, tau, condition=None):
        if tau.ndim == 1:
            tau = tau.repeat(z0.shape[0], 1)
        if bool(torch.allclose(tau, tau[0].expand_as(tau))):
            if self.model_type == "condition_aware_ode":
                self.ode_func.set_condition(condition)
            return odeint(self.ode_func, z0, tau[0], method="rk4")

        tau_cpu = tau.detach().cpu()
        unique_tau, inverse = torch.unique(tau_cpu, dim=0, return_inverse=True)
        steps = tau.shape[1]
        latent_dim = z0.shape[1]
        z_traj = torch.empty(steps, z0.shape[0], latent_dim, device=z0.device, dtype=z0.dtype)
        for group_id in range(unique_tau.shape[0]):
            member_positions = torch.nonzero(inverse == group_id, as_tuple=False).reshape(-1)
            idx = member_positions.to(device=z0.device)
            t_group = unique_tau[group_id].to(device=z0.device, dtype=z0.dtype)
            z0_group = z0.index_select(0, idx)
            if self.model_type == "condition_aware_ode":
                self.ode_func.set_condition(condition.index_select(0, idx))
            z_group = odeint(self.ode_func, z0_group, t_group, method="rk4")
            z_traj[:, idx, :] = z_group
        return z_traj


def load_split_payload(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    split = json.loads(data["split_json"].item()) if "split_json" in data else {}
    return data, split


def make_loader(X, y, c, tau, batch_size: int, shuffle: bool, device: torch.device):
    tensors = (
        torch.tensor(X, dtype=torch.float32, device=device),
        torch.tensor(y, dtype=torch.float32, device=device),
        torch.tensor(c, dtype=torch.float32, device=device),
        torch.tensor(tau, dtype=torch.float32, device=device),
    )
    dataset = TensorDataset(*tensors)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(RANDOM_SEED)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


@torch.no_grad()
def predict(model: nn.Module, X, c, tau, batch_size: int, device: torch.device):
    model.eval()
    if len(X) == 0:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    loader = make_loader(X, np.zeros(len(X), dtype=np.float32), c, tau, batch_size, shuffle=False, device=device)
    preds = []
    latents = []
    for xb, _, cb, tb in loader:
        pred, z_traj = model(xb, cb, tb)
        preds.append(pred.detach().cpu().numpy())
        latents.append(z_traj[-1].detach().cpu().numpy())
    return np.concatenate(preds, axis=0), np.concatenate(latents, axis=0)


def evaluate_mae(model, X, y, c, tau, batch_size, device) -> float:
    preds, _ = predict(model, X, c, tau, batch_size, device)
    return compute_mae_rmse_r2(y, preds)["mae"]


def metadata_frame(meta, y_true, y_pred, split: dict, model_name: str) -> pd.DataFrame:
    meta = np.asarray(meta, dtype=object)
    df = pd.DataFrame(meta, columns=["bearing_id", "condition_id", "time_index"])
    df["time_index"] = df["time_index"].astype(int)
    df["normalized_rul"] = np.asarray(y_true, dtype=float)
    df["y_pred"] = np.asarray(y_pred, dtype=float)
    df["model"] = model_name
    df["protocol"] = split.get("protocol", "")
    df["split_name"] = split.get("split_name", "")
    return df.sort_values(["bearing_id", "time_index"]).reset_index(drop=True)


def train_one(seq_path: Path, model_type: str) -> None:
    set_seed(RANDOM_SEED)
    data, split = load_split_payload(seq_path)
    split_name = split.get("split_name", seq_path.stem.replace("_k10", ""))
    protocol = split.get("protocol", "")

    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)
    c_train = (data["c_norm_train"] if "c_norm_train" in data else data["c_train"]).astype(np.float32)
    tau_train = data["tau_train"].astype(np.float32)
    X_val = data["X_val"].astype(np.float32)
    y_val = data["y_val"].astype(np.float32)
    c_val = (data["c_norm_val"] if "c_norm_val" in data else data["c_val"]).astype(np.float32)
    tau_val = data["tau_val"].astype(np.float32)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)
    c_test = (data["c_norm_test"] if "c_norm_test" in data else data["c_test"]).astype(np.float32)
    tau_test = data["tau_test"].astype(np.float32)

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise RuntimeError(f"Sequence file has an empty train/val/test subset: {seq_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[-1]
    latent_dim = 16
    lr = 1e-3
    batch_size = 64
    epochs = 100
    patience = 10
    smooth_weight = 1e-4

    model = LatentODERegressor(input_dim, latent_dim, model_type=model_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    train_loader = make_loader(X_train, y_train, c_train, tau_train, batch_size, shuffle=True, device=device)

    best_state = copy.deepcopy(model.state_dict())
    best_val_mae = float("inf")
    best_epoch = 0
    bad_epochs = 0

    print(f"Training {model_type} on {split_name} using {device}")
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb, cb, tb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred, z_traj = model(xb, cb, tb)
            pred_loss = loss_fn(pred, yb)
            smooth_loss = torch.mean((z_traj[1:] - z_traj[:-1]) ** 2)
            loss = pred_loss + smooth_weight * smooth_loss
            loss.backward()
            optimizer.step()

        val_mae = evaluate_mae(model, X_val, y_val, c_val, tau_val, batch_size, device)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    y_pred, z_test = predict(model, X_test, c_test, tau_test, batch_size, device)
    metrics = compute_mae_rmse_r2(y_test, y_pred)

    checkpoint_dir = ensure_dir("results/checkpoints")
    checkpoint_path = checkpoint_dir / f"{split_name}_{model_type}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "split": split,
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "model_type": model_type,
            "best_epoch": best_epoch,
            "best_val_mae": best_val_mae,
            "feature_names": data["feature_names"],
        },
        checkpoint_path,
    )

    pred_dir = ensure_dir("results/predictions")
    pred_path = pred_dir / f"{split_name}_{model_type}.csv"
    metadata_frame(data["meta_test"], y_test, y_pred, split, model_type).to_csv(pred_path, index=False)

    meta_test = np.asarray(data["meta_test"], dtype=object)
    latent_dir = ensure_dir("results/latent")
    latent_path = latent_dir / f"{split_name}_{model_type}_latent.npz"
    np.savez_compressed(
        latent_path,
        z_test=z_test.astype(np.float32),
        y_test=y_test.astype(np.float32),
        y_pred=y_pred.astype(np.float32),
        bearing_id=meta_test[:, 0].astype(str),
        condition_id=meta_test[:, 1].astype(str),
        time_index=meta_test[:, 2].astype(int),
    )

    best_params = {
        "latent_dim": latent_dim,
        "conditioning": "film_encoder_ode_decoder" if model_type == "condition_aware_ode" else "none",
        "learning_rate": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "patience": patience,
        "smooth_weight": smooth_weight,
        "ode_method": "rk4",
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
    }
    result_row = {
        "protocol": protocol,
        "split_name": split_name,
        "model": model_type,
        "input_type": "feature_sequence_continuous_time",
        "best_params": json.dumps(best_params, sort_keys=True),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "num_train_bearings": len(split.get("train_bearings", [])),
        "num_val_bearings": len(split.get("val_bearings", [])),
        "num_test_bearings": len(split.get("test_bearings", [])),
    }
    append_result_csv("results/tables/all_results.csv", {k: result_row[k] for k in RESULT_COLUMNS})
    print(f"{split_name} {model_type}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}")


def sequence_paths_from_args(seq: str | None, seq_dir: str | None) -> list[Path]:
    if seq:
        return [Path(seq)]
    if seq_dir:
        paths = sorted(Path(seq_dir).glob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"No sequence npz files found in {seq_dir}")
        return paths
    raise ValueError("Provide either --seq or --seq_dir")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Latent Neural ODE models.")
    parser.add_argument("--seq")
    parser.add_argument("--seq_dir")
    parser.add_argument("--model", choices=["latent_ode", "condition_aware_ode"], required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    for seq_path in sequence_paths_from_args(args.seq, args.seq_dir):
        train_one(seq_path, args.model)


if __name__ == "__main__":
    main()
