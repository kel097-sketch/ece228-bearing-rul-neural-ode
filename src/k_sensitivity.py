import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from make_sequences import make_sequence_file
from train_lstm import LSTMRegressor
from train_ode import LatentODERegressor
from train_tcn import TCNRegressor
from train_transformer import TransformerRegressor
from utils import compute_mae_rmse_r2, ensure_dir, load_split, set_seed


def tensor_loader(arrays: tuple[np.ndarray, ...], batch_size: int, shuffle: bool, device: torch.device, seed: int):
    tensors = tuple(torch.tensor(array, dtype=torch.float32, device=device) for array in arrays)
    dataset = TensorDataset(*tensors)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


@torch.no_grad()
def predict_discrete(model, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    for start in range(0, len(X), batch_size):
        xb = torch.tensor(X[start : start + batch_size], dtype=torch.float32, device=device)
        preds.append(model(xb).detach().cpu().numpy())
    return np.concatenate(preds) if preds else np.asarray([], dtype=np.float32)


@torch.no_grad()
def predict_ode(model, X: np.ndarray, c: np.ndarray, tau: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    for start in range(0, len(X), batch_size):
        xb = torch.tensor(X[start : start + batch_size], dtype=torch.float32, device=device)
        cb = torch.tensor(c[start : start + batch_size], dtype=torch.float32, device=device)
        tb = torch.tensor(tau[start : start + batch_size], dtype=torch.float32, device=device)
        pred, _ = model(xb, cb, tb)
        preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds) if preds else np.asarray([], dtype=np.float32)


def train_discrete_model(model, data: dict, args, device: torch.device, seed: int) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()
    train_loader = tensor_loader(
        (data["X_train"], data["y_train"]), args.batch_size, True, device, seed
    )
    best_state = copy.deepcopy(model.state_dict())
    best_val_mae = float("inf")
    best_epoch = 0
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        val_pred = predict_discrete(model, data["X_val"], args.batch_size, device)
        val_mae = compute_mae_rmse_r2(data["y_val"], val_pred)["mae"]
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break
    model.load_state_dict(best_state)
    pred = predict_discrete(model, data["X_test"], args.batch_size, device)
    metrics = compute_mae_rmse_r2(data["y_test"], pred)
    metrics["best_epoch"] = best_epoch
    metrics["best_val_mae"] = best_val_mae
    return metrics


def train_ode_model(model, data: dict, args, device: torch.device, seed: int) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()
    train_loader = tensor_loader(
        (data["X_train"], data["y_train"], data["c_train"], data["tau_train"]),
        args.batch_size,
        True,
        device,
        seed,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_val_mae = float("inf")
    best_epoch = 0
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb, cb, tb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred, z_traj = model(xb, cb, tb)
            pred_loss = loss_fn(pred, yb)
            smooth_loss = torch.mean((z_traj[1:] - z_traj[:-1]) ** 2)
            loss = pred_loss + args.smooth_weight * smooth_loss
            loss.backward()
            optimizer.step()
        val_pred = predict_ode(
            model, data["X_val"], data["c_val"], data["tau_val"], args.batch_size, device
        )
        val_mae = compute_mae_rmse_r2(data["y_val"], val_pred)["mae"]
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break
    model.load_state_dict(best_state)
    pred = predict_ode(
        model, data["X_test"], data["c_test"], data["tau_test"], args.batch_size, device
    )
    metrics = compute_mae_rmse_r2(data["y_test"], pred)
    metrics["best_epoch"] = best_epoch
    metrics["best_val_mae"] = best_val_mae
    return metrics


def load_sequence(path: Path) -> tuple[dict, dict]:
    npz = np.load(path, allow_pickle=True)
    split = json.loads(npz["split_json"].item()) if "split_json" in npz else {}
    data = {
        "X_train": npz["X_train"].astype(np.float32),
        "y_train": npz["y_train"].astype(np.float32),
        "X_val": npz["X_val"].astype(np.float32),
        "y_val": npz["y_val"].astype(np.float32),
        "X_test": npz["X_test"].astype(np.float32),
        "y_test": npz["y_test"].astype(np.float32),
        "c_train": (npz["c_norm_train"] if "c_norm_train" in npz else npz["c_train"]).astype(np.float32),
        "c_val": (npz["c_norm_val"] if "c_norm_val" in npz else npz["c_val"]).astype(np.float32),
        "c_test": (npz["c_norm_test"] if "c_norm_test" in npz else npz["c_test"]).astype(np.float32),
        "tau_train": npz["tau_train"].astype(np.float32),
        "tau_val": npz["tau_val"].astype(np.float32),
        "tau_test": npz["tau_test"].astype(np.float32),
    }
    return data, split


def model_instance(model_name: str, input_dim: int):
    if model_name == "LSTM":
        return LSTMRegressor(input_dim)
    if model_name == "TCN":
        return TCNRegressor(input_dim)
    if model_name == "Transformer":
        return TransformerRegressor(input_dim)
    if model_name in {"latent_ode", "condition_aware_ode"}:
        return LatentODERegressor(input_dim, model_type=model_name)
    raise ValueError(f"Unsupported model: {model_name}")


def run_k_sensitivity(args) -> pd.DataFrame:
    features_df = pd.read_csv(args.features)
    split_paths = []
    for split_path in sorted(Path(args.split_dir).glob("*.json")):
        split = load_split(split_path)
        if split.get("protocol") == args.protocol:
            split_paths.append(split_path)
    if not split_paths:
        raise FileNotFoundError(f"No splits found for protocol={args.protocol}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    sequence_root = ensure_dir(args.sequence_root)
    for k in args.k_values:
        k_dir = ensure_dir(sequence_root / f"k{k}")
        for split_path in split_paths:
            seq_path = k_dir / f"{load_split(split_path)['split_name']}_k{k}.npz"
            if not seq_path.exists() or args.rebuild_sequences:
                seq_path = make_sequence_file(features_df, split_path, k, k_dir)
            data, split = load_sequence(seq_path)
            if len(data["X_train"]) == 0 or len(data["X_val"]) == 0 or len(data["X_test"]) == 0:
                print(f"Skipping empty sequence set: {seq_path}")
                continue
            for model_name in args.models:
                seed = args.seed + 1000 * k + sum(ord(ch) for ch in split["split_name"] + model_name)
                set_seed(seed)
                model = model_instance(model_name, data["X_train"].shape[-1]).to(device)
                print(f"K sensitivity: k={k}, split={split['split_name']}, model={model_name}, device={device}")
                if model_name in {"latent_ode", "condition_aware_ode"}:
                    metrics = train_ode_model(model, data, args, device, seed)
                else:
                    metrics = train_discrete_model(model, data, args, device, seed)
                rows.append(
                    {
                        "k": k,
                        "protocol": split.get("protocol", ""),
                        "split_name": split.get("split_name", split_path.stem),
                        "model": model_name,
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "r2": metrics["r2"],
                        "best_epoch": metrics["best_epoch"],
                        "best_val_mae": metrics["best_val_mae"],
                    }
                )
    results = pd.DataFrame(rows)
    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    results.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(results)} rows)")
    if not results.empty:
        avg = (
            results.groupby(["k", "model"], as_index=False)
            .agg(
                mae=("mae", "mean"),
                rmse=("rmse", "mean"),
                r2=("r2", "mean"),
                mae_std=("mae", "std"),
                num_splits=("split_name", "nunique"),
            )
            .sort_values(["k", "mae"])
        )
        avg_path = out_path.with_name("k_sensitivity_average_results.csv")
        avg.to_csv(avg_path, index=False)
        print(f"Saved {avg_path} ({len(avg)} rows)")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Window-length sensitivity for sequence RUL models.")
    parser.add_argument("--features", default="processed/features.csv")
    parser.add_argument("--split_dir", default="processed/splits")
    parser.add_argument("--sequence_root", type=Path, default=Path("processed/sequences_k_sensitivity"))
    parser.add_argument("--out", default="results/tables/k_sensitivity_results.csv")
    parser.add_argument("--protocol", default="cross_condition")
    parser.add_argument("--k_values", nargs="+", type=int, default=[5, 10, 20, 30])
    parser.add_argument("--models", nargs="+", default=["LSTM", "TCN", "Transformer", "latent_ode"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--smooth_weight", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild_sequences", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_k_sensitivity(args)


if __name__ == "__main__":
    main()
