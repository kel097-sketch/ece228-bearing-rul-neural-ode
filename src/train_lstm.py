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


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        final_hidden = h_n[-1]
        return self.head(final_hidden).squeeze(-1)


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


def evaluate_mae(model, X, y, batch_size, device) -> float:
    preds = predict(model, X, batch_size, device)
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
    input_dim = X_train.shape[-1]
    hidden_dim = 64
    num_layers = 1
    dropout = 0.1
    lr = 1e-3
    batch_size = 64
    epochs = 100
    patience = 10

    model = LSTMRegressor(input_dim, hidden_dim, num_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    train_loader = make_loader(X_train, y_train, batch_size, shuffle=True, device=device)

    best_state = copy.deepcopy(model.state_dict())
    best_val_mae = float("inf")
    best_epoch = 0
    bad_epochs = 0

    print(f"Training LSTM on {split_name} using {device}")
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        val_mae = evaluate_mae(model, X_val, y_val, batch_size, device)
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

    checkpoint_dir = ensure_dir("results/checkpoints")
    checkpoint_path = checkpoint_dir / f"{split_name}_lstm.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "split": split,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "best_epoch": best_epoch,
            "best_val_mae": best_val_mae,
            "feature_names": data["feature_names"],
        },
        checkpoint_path,
    )

    pred_dir = ensure_dir("results/predictions")
    pred_path = pred_dir / f"{split_name}_LSTM.csv"
    metadata_frame(data["meta_test"], y_test, y_pred, split, "LSTM").to_csv(pred_path, index=False)

    best_params = {
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
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
        "model": "LSTM",
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
    print(f"{split_name} LSTM: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}")


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
    parser = argparse.ArgumentParser(description="Train LSTM sequence baseline.")
    parser.add_argument("--seq")
    parser.add_argument("--seq_dir")
    return parser.parse_args()


def main():
    args = parse_args()
    for seq_path in sequence_paths_from_args(args.seq, args.seq_dir):
        train_one(seq_path)


if __name__ == "__main__":
    main()
