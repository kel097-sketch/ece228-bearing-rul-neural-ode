import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)


class TCNRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        levels: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        blocks = []
        in_channels = input_dim
        for level in range(levels):
            dilation = 2**level
            blocks.append(TemporalBlock(in_channels, hidden_dim, kernel_size, dilation, dropout))
            in_channels = hidden_dim
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        h = self.tcn(x).transpose(1, 2)
        return self.head(h[:, -1, :]).squeeze(-1)


def load_split_payload(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    split = json.loads(data["split_json"].item()) if "split_json" in data else {}
    return data, split


def make_loader(X, y, batch_size: int, shuffle: bool, device: torch.device):
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
    dataset = TensorDataset(X_tensor, y_tensor)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(RANDOM_SEED)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


@torch.no_grad()
def predict(model: nn.Module, X, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    if len(X) == 0:
        return np.asarray([], dtype=np.float32)
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)
    preds = []
    for (xb,) in loader:
        preds.append(model(xb).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


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


def train_one(seq_path: Path) -> None:
    set_seed(RANDOM_SEED)
    data, split = load_split_payload(seq_path)
    split_name = split.get("split_name", seq_path.stem.replace("_k10", ""))
    protocol = split.get("protocol", "")

    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)
    X_val = data["X_val"].astype(np.float32)
    y_val = data["y_val"].astype(np.float32)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)
    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise RuntimeError(f"Sequence file has an empty train/val/test subset: {seq_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = 64
    levels = 3
    kernel_size = 3
    dropout = 0.1
    lr = 1e-3
    batch_size = 64
    epochs = 100
    patience = 10

    model = TCNRegressor(X_train.shape[-1], hidden_dim, levels, kernel_size, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    train_loader = make_loader(X_train, y_train, batch_size, shuffle=True, device=device)

    best_state = copy.deepcopy(model.state_dict())
    best_val_mae = float("inf")
    best_epoch = 0
    bad_epochs = 0
    print(f"Training TCN on {split_name} using {device}")
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        val_pred = predict(model, X_val, batch_size, device)
        val_mae = compute_mae_rmse_r2(y_val, val_pred)["mae"]
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
    y_pred = predict(model, X_test, batch_size, device)
    metrics = compute_mae_rmse_r2(y_test, y_pred)

    checkpoint_path = ensure_dir("results/checkpoints") / f"{split_name}_tcn.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "split": split,
            "input_dim": X_train.shape[-1],
            "hidden_dim": hidden_dim,
            "levels": levels,
            "kernel_size": kernel_size,
            "dropout": dropout,
            "best_epoch": best_epoch,
            "best_val_mae": best_val_mae,
            "feature_names": data["feature_names"],
        },
        checkpoint_path,
    )
    pred_path = ensure_dir("results/predictions") / f"{split_name}_TCN.csv"
    metadata_frame(data["meta_test"], y_test, y_pred, split, "TCN").to_csv(pred_path, index=False)

    best_params = {
        "hidden_dim": hidden_dim,
        "levels": levels,
        "kernel_size": kernel_size,
        "dropout": dropout,
        "learning_rate": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "patience": patience,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
    }
    result_row = {
        "protocol": protocol,
        "split_name": split_name,
        "model": "TCN",
        "input_type": "feature_sequence",
        "best_params": json.dumps(best_params, sort_keys=True),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "num_train_bearings": len(split.get("train_bearings", [])),
        "num_val_bearings": len(split.get("val_bearings", [])),
        "num_test_bearings": len(split.get("test_bearings", [])),
    }
    append_result_csv("results/tables/all_results.csv", {k: result_row[k] for k in RESULT_COLUMNS})
    print(f"{split_name} TCN: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}")


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
    parser = argparse.ArgumentParser(description="Train TCN sequence baseline.")
    parser.add_argument("--seq")
    parser.add_argument("--seq_dir")
    return parser.parse_args()


def main():
    args = parse_args()
    for seq_path in sequence_paths_from_args(args.seq, args.seq_dir):
        train_one(seq_path)


if __name__ == "__main__":
    main()
