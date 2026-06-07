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
from train_lstm import LSTMRegressor
from train_ode import LatentODERegressor
from train_tcn import TCNRegressor
from train_transformer import TransformerRegressor
from utils import compute_mae_rmse_r2, ensure_dir, set_seed


def stable_seed(text: str, offset: int = 0) -> int:
    value = sum((i + 1) * ord(ch) for i, ch in enumerate(text))
    return RANDOM_SEED + offset + value % 100000


def make_mask_bank(k: int, keep_count: int, rng: np.random.Generator, bank_size: int) -> np.ndarray:
    if keep_count >= k:
        return np.arange(k, dtype=np.int64)[None, :]

    interior = np.arange(1, k - 1, dtype=np.int64)
    masks = set()
    attempts = 0
    max_attempts = max(100, bank_size * 20)
    while len(masks) < bank_size and attempts < max_attempts:
        chosen = rng.choice(interior, size=keep_count - 2, replace=False)
        mask = tuple(sorted([0, *chosen.tolist(), k - 1]))
        masks.add(mask)
        attempts += 1
    return np.asarray(sorted(masks), dtype=np.int64)


def sparsify_sequences(
    X: np.ndarray,
    keep_ratio: float,
    seed: int,
    mask_bank_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    n, k, _ = X.shape
    keep_count = k if keep_ratio >= 1.0 else max(2, int(round(k * keep_ratio)))
    keep_count = min(k, keep_count)
    rng = np.random.default_rng(seed)
    mask_bank = make_mask_bank(k, keep_count, rng, mask_bank_size)
    assignments = rng.integers(0, len(mask_bank), size=n)
    idx = mask_bank[assignments]
    X_sparse = X[np.arange(n)[:, None], idx]
    tau_sparse = (idx / max(k - 1, 1)).astype(np.float32)
    return X_sparse.astype(np.float32), tau_sparse.astype(np.float32)


def interpolate_sparse_to_full(X_sparse: np.ndarray, tau_sparse: np.ndarray, full_k: int) -> np.ndarray:
    if X_sparse.shape[1] == full_k and np.allclose(tau_sparse[0], np.linspace(0.0, 1.0, full_k)):
        return X_sparse.astype(np.float32)
    target_tau = np.linspace(0.0, 1.0, full_k, dtype=np.float32)
    n, _, feature_dim = X_sparse.shape
    out = np.empty((n, full_k, feature_dim), dtype=np.float32)
    for i in range(n):
        for j in range(feature_dim):
            out[i, :, j] = np.interp(target_tau, tau_sparse[i], X_sparse[i, :, j])
    return out


def tensor_loader(arrays: tuple[np.ndarray, ...], batch_size: int, shuffle: bool, device: torch.device):
    tensors = tuple(torch.tensor(array, dtype=torch.float32, device=device) for array in arrays)
    dataset = TensorDataset(*tensors)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(RANDOM_SEED)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


@torch.no_grad()
def predict_lstm(model, X, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    loader = tensor_loader((X,), batch_size=batch_size, shuffle=False, device=device)
    preds = []
    for (xb,) in loader:
        preds.append(model(xb).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


@torch.no_grad()
def predict_ode(model, X, c, tau, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    loader = tensor_loader((X, c, tau), batch_size=batch_size, shuffle=False, device=device)
    preds = []
    for xb, cb, tb in loader:
        pred, _ = model(xb, cb, tb)
        preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def train_lstm_sparse(data: dict, keep_ratio: float, args, device: torch.device) -> dict:
    X_train_s, tau_train_s = sparsify_sequences(
        data["X_train"], keep_ratio, args.seed + 11, args.mask_bank_size
    )
    X_val_s, tau_val_s = sparsify_sequences(
        data["X_val"], keep_ratio, args.seed + 12, args.mask_bank_size
    )
    X_test_s, tau_test_s = sparsify_sequences(
        data["X_test"], keep_ratio, args.seed + 13, args.mask_bank_size
    )
    X_train = interpolate_sparse_to_full(X_train_s, tau_train_s, data["X_train"].shape[1])
    X_val = interpolate_sparse_to_full(X_val_s, tau_val_s, data["X_val"].shape[1])
    X_test = interpolate_sparse_to_full(X_test_s, tau_test_s, data["X_test"].shape[1])

    model = LSTMRegressor(data["X_train"].shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()
    train_loader = tensor_loader((X_train, data["y_train"]), args.batch_size, True, device)

    best_state = copy.deepcopy(model.state_dict())
    best_val_mae = float("inf")
    bad_epochs = 0
    for _ in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        val_pred = predict_lstm(model, X_val, args.batch_size, device)
        val_mae = compute_mae_rmse_r2(data["y_val"], val_pred)["mae"]
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    model.load_state_dict(best_state)
    y_pred = predict_lstm(model, X_test, args.batch_size, device)
    return compute_mae_rmse_r2(data["y_test"], y_pred)


def train_discrete_sparse(
    data: dict,
    keep_ratio: float,
    args,
    device: torch.device,
    model_factory,
    seed_offset: int,
) -> dict:
    X_train_s, tau_train_s = sparsify_sequences(
        data["X_train"], keep_ratio, args.seed + seed_offset + 1, args.mask_bank_size
    )
    X_val_s, tau_val_s = sparsify_sequences(
        data["X_val"], keep_ratio, args.seed + seed_offset + 2, args.mask_bank_size
    )
    X_test_s, tau_test_s = sparsify_sequences(
        data["X_test"], keep_ratio, args.seed + seed_offset + 3, args.mask_bank_size
    )
    X_train = interpolate_sparse_to_full(X_train_s, tau_train_s, data["X_train"].shape[1])
    X_val = interpolate_sparse_to_full(X_val_s, tau_val_s, data["X_val"].shape[1])
    X_test = interpolate_sparse_to_full(X_test_s, tau_test_s, data["X_test"].shape[1])

    model = model_factory(data["X_train"].shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()
    train_loader = tensor_loader((X_train, data["y_train"]), args.batch_size, True, device)

    best_state = copy.deepcopy(model.state_dict())
    best_val_mae = float("inf")
    bad_epochs = 0
    for _ in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        val_pred = predict_lstm(model, X_val, args.batch_size, device)
        val_mae = compute_mae_rmse_r2(data["y_val"], val_pred)["mae"]
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    model.load_state_dict(best_state)
    y_pred = predict_lstm(model, X_test, args.batch_size, device)
    return compute_mae_rmse_r2(data["y_test"], y_pred)


def train_ode_sparse(data: dict, keep_ratio: float, args, device: torch.device) -> dict:
    X_train, tau_train = sparsify_sequences(data["X_train"], keep_ratio, args.seed + 21, args.mask_bank_size)
    X_val, tau_val = sparsify_sequences(data["X_val"], keep_ratio, args.seed + 22, args.mask_bank_size)
    X_test, tau_test = sparsify_sequences(data["X_test"], keep_ratio, args.seed + 23, args.mask_bank_size)
    c_train = data["c_train"]
    c_val = data["c_val"]
    c_test = data["c_test"]

    model = LatentODERegressor(data["X_train"].shape[-1], model_type="latent_ode").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()
    train_loader = tensor_loader((X_train, data["y_train"], c_train, tau_train), args.batch_size, True, device)

    best_state = copy.deepcopy(model.state_dict())
    best_val_mae = float("inf")
    bad_epochs = 0
    for _ in range(args.epochs):
        model.train()
        for xb, yb, cb, tb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred, z_traj = model(xb, cb, tb)
            pred_loss = loss_fn(pred, yb)
            smooth_loss = torch.mean((z_traj[1:] - z_traj[:-1]) ** 2)
            loss = pred_loss + args.smooth_weight * smooth_loss
            loss.backward()
            optimizer.step()
        val_pred = predict_ode(model, X_val, c_val, tau_val, args.batch_size, device)
        val_mae = compute_mae_rmse_r2(data["y_val"], val_pred)["mae"]
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    model.load_state_dict(best_state)
    y_pred = predict_ode(model, X_test, c_test, tau_test, args.batch_size, device)
    return compute_mae_rmse_r2(data["y_test"], y_pred)


def load_sequence(path: Path) -> tuple[dict, dict]:
    npz = np.load(path, allow_pickle=True)
    split = json.loads(npz["split_json"].item()) if "split_json" in npz else {}
    data = {
        "X_train": npz["X_train"].astype(np.float32),
        "y_train": npz["y_train"].astype(np.float32),
        "c_train": (npz["c_norm_train"] if "c_norm_train" in npz else npz["c_train"]).astype(np.float32),
        "X_val": npz["X_val"].astype(np.float32),
        "y_val": npz["y_val"].astype(np.float32),
        "c_val": (npz["c_norm_val"] if "c_norm_val" in npz else npz["c_val"]).astype(np.float32),
        "X_test": npz["X_test"].astype(np.float32),
        "y_test": npz["y_test"].astype(np.float32),
        "c_test": (npz["c_norm_test"] if "c_norm_test" in npz else npz["c_test"]).astype(np.float32),
    }
    return data, split


def run_sparse_experiment(args) -> pd.DataFrame:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_paths = sorted(Path(args.seq_dir).glob("*.npz"))
    rows = []

    for seq_path in seq_paths:
        data, split = load_sequence(seq_path)
        if split.get("protocol") != "cross_condition":
            continue
        split_name = split.get("split_name", seq_path.stem)
        if len(data["X_train"]) == 0 or len(data["X_val"]) == 0 or len(data["X_test"]) == 0:
            print(f"Skipping {split_name}; empty train/val/test sequence set.")
            continue
        for keep_ratio in args.keep_ratios:
            run_seed = stable_seed(split_name, int(round(keep_ratio * 1000)))
            args.seed = run_seed

            print(f"Training sparse LSTM: split={split_name}, keep_ratio={keep_ratio}, device={device}")
            set_seed(run_seed)
            metrics = train_lstm_sparse(data, keep_ratio, args, device)
            rows.append(
                {
                    "model": "LSTM",
                    "split_name": split_name,
                    "keep_ratio": keep_ratio,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "r2": metrics["r2"],
                }
            )

            for model_name, model_factory, seed_offset in [
                ("TCN", TCNRegressor, 100),
                ("Transformer", TransformerRegressor, 200),
            ]:
                print(
                    f"Training sparse {model_name}: "
                    f"split={split_name}, keep_ratio={keep_ratio}, device={device}"
                )
                set_seed(run_seed + seed_offset)
                metrics = train_discrete_sparse(
                    data, keep_ratio, args, device, model_factory, seed_offset
                )
                rows.append(
                    {
                        "model": model_name,
                        "split_name": split_name,
                        "keep_ratio": keep_ratio,
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "r2": metrics["r2"],
                    }
                )

            print(f"Training sparse latent_ode: split={split_name}, keep_ratio={keep_ratio}, device={device}")
            set_seed(run_seed)
            metrics = train_ode_sparse(data, keep_ratio, args, device)
            rows.append(
                {
                    "model": "latent_ode",
                    "split_name": split_name,
                    "keep_ratio": keep_ratio,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "r2": metrics["r2"],
                }
            )

    results = pd.DataFrame(rows, columns=["model", "split_name", "keep_ratio", "mae", "rmse", "r2"])
    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    results.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(results)} rows)")
    if not results.empty:
        summary = (
            results.groupby(["model", "keep_ratio"], as_index=False)[["mae", "rmse", "r2"]]
            .mean()
            .sort_values(["keep_ratio", "mae"], ascending=[False, True])
        )
        summary_path = out_path.with_name("sparse_observation_average_results.csv")
        summary.to_csv(summary_path, index=False)
        print(f"Saved {summary_path} ({len(summary)} rows)")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Sparse-observation robustness for LSTM and Latent ODE.")
    parser.add_argument("--seq_dir", default="processed/sequences")
    parser.add_argument("--out", default="results/tables/sparse_observation_results.csv")
    parser.add_argument("--keep_ratios", nargs="+", type=float, default=[1.0, 0.7, 0.5, 0.3])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--smooth_weight", type=float, default=1e-4)
    parser.add_argument("--mask_bank_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    run_sparse_experiment(args)


if __name__ == "__main__":
    main()
