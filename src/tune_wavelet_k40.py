import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from make_sequences import make_sequence_file
from strict_wavelet_experiment import (
    aggregate_bearing_metrics,
    per_bearing_metrics,
    prediction_frame,
    summarize_final,
)
from train_lstm import LSTMRegressor
from train_ode import LatentODERegressor
from train_tcn import TCNRegressor
from train_transformer import TransformerRegressor
from utils import ensure_dir, get_feature_columns, load_split, set_seed


MODEL_ORDER = ["Transformer", "LSTM", "TCN", "latent_ode", "Ridge"]
MODEL_COLORS = {
    "Ridge": "#8c8c8c",
    "LSTM": "#4C78A8",
    "TCN": "#00A676",
    "Transformer": "#0B63B6",
    "latent_ode": "#8064A2",
}


def cross_split_paths(split_dir: Path) -> list[Path]:
    paths = []
    for path in sorted(split_dir.glob("*.json")):
        split = load_split(path)
        if split.get("protocol") == "cross_condition":
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No cross_condition splits found in {split_dir}")
    return paths


def split_df(features_df: pd.DataFrame, bearings: list[str]) -> pd.DataFrame:
    return features_df[features_df["bearing_id"].isin(bearings)].copy()


def load_sequence(path: Path) -> tuple[dict, dict]:
    npz = np.load(path, allow_pickle=True)
    split = json.loads(npz["split_json"].item())
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
        "meta_val": npz["meta_val"],
        "meta_test": npz["meta_test"],
        "feature_names": npz["feature_names"],
    }
    return data, split


def tensor_loader(arrays: tuple[np.ndarray, ...], batch_size: int, shuffle: bool, device: torch.device, seed: int):
    tensors = tuple(torch.tensor(array, dtype=torch.float32, device=device) for array in arrays)
    dataset = TensorDataset(*tensors)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


@torch.no_grad()
def predict_discrete(model: nn.Module, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    for start in range(0, len(X), batch_size):
        xb = torch.tensor(X[start : start + batch_size], dtype=torch.float32, device=device)
        preds.append(model(xb).detach().cpu().numpy())
    return np.concatenate(preds) if preds else np.asarray([], dtype=np.float32)


@torch.no_grad()
def predict_ode(
    model: nn.Module,
    X: np.ndarray,
    c: np.ndarray,
    tau: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    preds = []
    for start in range(0, len(X), batch_size):
        xb = torch.tensor(X[start : start + batch_size], dtype=torch.float32, device=device)
        cb = torch.tensor(c[start : start + batch_size], dtype=torch.float32, device=device)
        tb = torch.tensor(tau[start : start + batch_size], dtype=torch.float32, device=device)
        pred, _ = model(xb, cb, tb)
        preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds) if preds else np.asarray([], dtype=np.float32)


def build_model(model_name: str, input_dim: int, params: dict) -> nn.Module:
    if model_name == "LSTM":
        return LSTMRegressor(
            input_dim=input_dim,
            hidden_dim=int(params["hidden_dim"]),
            num_layers=int(params["num_layers"]),
            dropout=float(params["dropout"]),
        )
    if model_name == "TCN":
        return TCNRegressor(
            input_dim=input_dim,
            hidden_dim=int(params["hidden_dim"]),
            levels=int(params["levels"]),
            kernel_size=int(params["kernel_size"]),
            dropout=float(params["dropout"]),
        )
    if model_name == "Transformer":
        return TransformerRegressor(
            input_dim=input_dim,
            d_model=int(params["d_model"]),
            nhead=int(params["nhead"]),
            num_layers=int(params["num_layers"]),
            dim_feedforward=int(params["dim_feedforward"]),
            dropout=float(params["dropout"]),
        )
    if model_name == "latent_ode":
        return LatentODERegressor(input_dim=input_dim, latent_dim=int(params["latent_dim"]), model_type="latent_ode")
    raise ValueError(model_name)


def model_grid() -> dict[str, list[dict]]:
    return {
        "LSTM": [
            {"hidden_dim": 32, "num_layers": 1, "dropout": 0.0, "lr": 1e-3, "weight_decay": 1e-4},
            {"hidden_dim": 64, "num_layers": 1, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4},
            {"hidden_dim": 128, "num_layers": 1, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4},
            {"hidden_dim": 64, "num_layers": 2, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4},
            {"hidden_dim": 128, "num_layers": 2, "dropout": 0.2, "lr": 5e-4, "weight_decay": 1e-4},
            {"hidden_dim": 64, "num_layers": 2, "dropout": 0.2, "lr": 5e-4, "weight_decay": 1e-3},
        ],
        "TCN": [
            {"hidden_dim": 32, "levels": 2, "kernel_size": 3, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4},
            {"hidden_dim": 64, "levels": 2, "kernel_size": 3, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4},
            {"hidden_dim": 64, "levels": 3, "kernel_size": 3, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4},
            {"hidden_dim": 64, "levels": 3, "kernel_size": 5, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4},
            {"hidden_dim": 128, "levels": 3, "kernel_size": 3, "dropout": 0.1, "lr": 5e-4, "weight_decay": 1e-4},
            {"hidden_dim": 128, "levels": 3, "kernel_size": 5, "dropout": 0.2, "lr": 5e-4, "weight_decay": 1e-4},
        ],
        "Transformer": [
            {"d_model": 32, "nhead": 2, "num_layers": 1, "dim_feedforward": 64, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4},
            {"d_model": 64, "nhead": 4, "num_layers": 1, "dim_feedforward": 128, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4},
            {"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128, "dropout": 0.1, "lr": 5e-4, "weight_decay": 1e-4},
            {"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 256, "dropout": 0.2, "lr": 5e-4, "weight_decay": 1e-4},
            {"d_model": 128, "nhead": 4, "num_layers": 2, "dim_feedforward": 256, "dropout": 0.2, "lr": 5e-4, "weight_decay": 1e-4},
            {"d_model": 32, "nhead": 2, "num_layers": 2, "dim_feedforward": 128, "dropout": 0.2, "lr": 5e-4, "weight_decay": 1e-3},
        ],
        "latent_ode": [
            {"latent_dim": 8, "lr": 1e-3, "weight_decay": 1e-4, "smooth_weight": 1e-4},
            {"latent_dim": 16, "lr": 1e-3, "weight_decay": 1e-4, "smooth_weight": 1e-4},
            {"latent_dim": 32, "lr": 5e-4, "weight_decay": 1e-4, "smooth_weight": 1e-4},
            {"latent_dim": 32, "lr": 1e-3, "weight_decay": 1e-4, "smooth_weight": 5e-4},
        ],
    }


def train_one_config(
    model_name: str,
    params: dict,
    data: dict,
    args,
    device: torch.device,
    seed: int,
) -> dict:
    set_seed(seed)
    model = build_model(model_name, data["X_train"].shape[-1], params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params["lr"]), weight_decay=float(params["weight_decay"]))
    loss_fn = nn.MSELoss()
    batch_size = int(args.batch_size)
    if model_name == "latent_ode":
        train_loader = tensor_loader(
            (data["X_train"], data["y_train"], data["c_train"], data["tau_train"]),
            batch_size,
            True,
            device,
            seed,
        )
    else:
        train_loader = tensor_loader((data["X_train"], data["y_train"]), batch_size, True, device, seed)

    best_state = copy.deepcopy(model.state_dict())
    best_val = None
    best_val_mae = float("inf")
    best_epoch = 0
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if model_name == "latent_ode":
                xb, yb, cb, tb = batch
                pred, z_traj = model(xb, cb, tb)
                smooth_loss = torch.mean((z_traj[1:] - z_traj[:-1]) ** 2)
                loss = loss_fn(pred, yb) + float(params.get("smooth_weight", args.smooth_weight)) * smooth_loss
            else:
                xb, yb = batch
                loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        if model_name == "latent_ode":
            val_pred = predict_ode(model, data["X_val"], data["c_val"], data["tau_val"], batch_size, device)
        else:
            val_pred = predict_discrete(model, data["X_val"], batch_size, device)
        val_pred = np.clip(val_pred, 0.0, 1.0)
        val_metrics = aggregate_bearing_metrics(
            per_bearing_metrics(data["meta_val"], data["y_val"], val_pred, args.epsilon)
        )
        val_mae = val_metrics["mae"]
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val = val_metrics
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    model.load_state_dict(best_state)
    if model_name == "latent_ode":
        val_pred = predict_ode(model, data["X_val"], data["c_val"], data["tau_val"], batch_size, device)
        test_pred = predict_ode(model, data["X_test"], data["c_test"], data["tau_test"], batch_size, device)
    else:
        val_pred = predict_discrete(model, data["X_val"], batch_size, device)
        test_pred = predict_discrete(model, data["X_test"], batch_size, device)
    val_pred = np.clip(val_pred, 0.0, 1.0)
    test_pred = np.clip(test_pred, 0.0, 1.0)
    val_metrics = aggregate_bearing_metrics(per_bearing_metrics(data["meta_val"], data["y_val"], val_pred, args.epsilon))
    test_per_bearing = per_bearing_metrics(data["meta_test"], data["y_test"], test_pred, args.epsilon)
    return {
        "model_state": copy.deepcopy(model.state_dict()),
        "val_metrics": val_metrics,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "best_val_metrics_during_training": best_val,
        "test_pred": test_pred,
        "test_per_bearing": test_per_bearing,
    }


def train_ridge(features_df: pd.DataFrame, split: dict, feature_cols: list[str], args) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_df = split_df(features_df, split["train_bearings"])
    val_df = split_df(features_df, split["val_bearings"])
    test_df = split_df(features_df, split["test_bearings"])
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].to_numpy(dtype=np.float32))
    X_val = scaler.transform(val_df[feature_cols].to_numpy(dtype=np.float32))
    X_test = scaler.transform(test_df[feature_cols].to_numpy(dtype=np.float32))
    y_train = train_df["normalized_rul"].to_numpy(dtype=np.float32)
    y_val = val_df["normalized_rul"].to_numpy(dtype=np.float32)
    best = None
    trial_rows = []
    for alpha in args.ridge_alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        pred_val = np.clip(model.predict(X_val), 0.0, 1.0)
        val_metrics = aggregate_bearing_metrics(
            per_bearing_metrics(
                val_df[["bearing_id", "condition_id", "time_index"]].to_numpy(dtype=object),
                y_val,
                pred_val,
                args.epsilon,
            )
        )
        trial_rows.append({"alpha": alpha, **val_metrics})
        if best is None or val_metrics["mae"] < best["mae"]:
            best = {"alpha": alpha, "model": model, **val_metrics}

    y_test = test_df["normalized_rul"].to_numpy(dtype=np.float32)
    pred_test = np.clip(best["model"].predict(X_test), 0.0, 1.0)
    test_meta = test_df[["bearing_id", "condition_id", "time_index"]].to_numpy(dtype=object)
    pred = prediction_frame(test_meta, y_test, pred_test, split, "Ridge", "wavelet_only", None)
    per_bearing = per_bearing_metrics(test_meta, y_test, pred_test, args.epsilon)
    per_bearing["model"] = "Ridge"
    per_bearing["feature_setting"] = "wavelet_only"
    per_bearing["K"] = "NA"
    per_bearing["protocol"] = split["protocol"]
    per_bearing["split_name"] = split["split_name"]
    per_bearing["best_params"] = json.dumps({"alpha": best["alpha"]}, sort_keys=True)
    return pred, per_bearing, {"best": best, "trials": trial_rows}


def run(args) -> None:
    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    fig_dir = ensure_dir(args.fig_dir)
    pred_dir = ensure_dir(args.pred_dir)
    seq_dir = ensure_dir(args.seq_dir)
    ckpt_dir = ensure_dir(args.ckpt_dir)
    features_df = pd.read_csv(args.features)
    feature_cols = get_feature_columns(features_df)
    if len(feature_cols) != 72:
        raise RuntimeError(f"Expected 72 wavelet-only features, found {len(feature_cols)}")
    split_paths = cross_split_paths(Path(args.split_dir))
    grid = model_grid()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device={device}, K={args.k}, feature_dim={len(feature_cols)}", flush=True)

    all_trials = []
    best_rows = []
    all_per_bearing = []
    all_predictions = []
    for split_path in split_paths:
        split = load_split(split_path)
        print(f"\n=== Split {split['split_name']} ===", flush=True)
        seq_path = seq_dir / f"{split['split_name']}_k{args.k}.npz"
        if args.rebuild_sequences or not seq_path.exists():
            seq_path = make_sequence_file(features_df, split_path, args.k, seq_dir)
        data, _ = load_sequence(seq_path)

        ridge_pred, ridge_metrics, ridge_info = train_ridge(features_df, split, feature_cols, args)
        all_predictions.append(ridge_pred)
        all_per_bearing.append(ridge_metrics)
        for trial in ridge_info["trials"]:
            all_trials.append(
                {
                    "split_name": split["split_name"],
                    "model": "Ridge",
                    "config_id": f"ridge_alpha_{trial['alpha']}",
                    "params": json.dumps({"alpha": trial["alpha"]}, sort_keys=True),
                    "val_mae": trial["mae"],
                    "val_rmse": trial["rmse"],
                    "val_spearman": trial["spearman"],
                    "best_epoch": None,
                }
            )
        best_rows.append(
            {
                "split_name": split["split_name"],
                "model": "Ridge",
                "params": json.dumps({"alpha": ridge_info["best"]["alpha"]}, sort_keys=True),
                "val_mae": ridge_info["best"]["mae"],
                "best_epoch": None,
            }
        )

        for model_name in ["LSTM", "TCN", "Transformer", "latent_ode"]:
            best_result = None
            best_params = None
            best_config_id = None
            for config_id, params in enumerate(grid[model_name], start=1):
                seed = args.seed + sum(ord(ch) for ch in f"{split['split_name']}:{model_name}:{config_id}")
                print(f"Tuning {model_name} split={split['split_name']} config={config_id}/{len(grid[model_name])}", flush=True)
                result = train_one_config(model_name, params, data, args, device, seed)
                val = result["val_metrics"]
                all_trials.append(
                    {
                        "split_name": split["split_name"],
                        "model": model_name,
                        "config_id": config_id,
                        "params": json.dumps(params, sort_keys=True),
                        "val_mae": val["mae"],
                        "val_rmse": val["rmse"],
                        "val_spearman": val["spearman"],
                        "val_late_mae": val["late_mae"],
                        "best_epoch": result["best_epoch"],
                    }
                )
                if best_result is None or val["mae"] < best_result["val_metrics"]["mae"]:
                    best_result = result
                    best_params = params
                    best_config_id = config_id

            assert best_result is not None and best_params is not None
            print(
                f"Selected {model_name} config={best_config_id} val_MAE={best_result['val_metrics']['mae']:.4f}",
                flush=True,
            )
            torch.save(
                {
                    "model_name": model_name,
                    "params": best_params,
                    "model_state_dict": best_result["model_state"],
                    "split": split,
                    "K": args.k,
                    "feature_names": data["feature_names"],
                    "best_epoch": best_result["best_epoch"],
                },
                ckpt_dir / f"{split['split_name']}_{model_name}_K{args.k}_tuned.pt",
            )
            test_metrics = best_result["test_per_bearing"].copy()
            test_metrics["model"] = model_name
            test_metrics["feature_setting"] = "wavelet_only"
            test_metrics["K"] = args.k
            test_metrics["protocol"] = split["protocol"]
            test_metrics["split_name"] = split["split_name"]
            test_metrics["best_params"] = json.dumps(best_params, sort_keys=True)
            all_per_bearing.append(test_metrics)
            pred_frame = prediction_frame(
                data["meta_test"],
                data["y_test"],
                best_result["test_pred"],
                split,
                model_name,
                "wavelet_only",
                args.k,
            )
            pred_frame.to_csv(pred_dir / f"{split['split_name']}_{model_name}_K{args.k}_tuned_predictions.csv", index=False)
            all_predictions.append(pred_frame)
            best_rows.append(
                {
                    "split_name": split["split_name"],
                    "model": model_name,
                    "params": json.dumps(best_params, sort_keys=True),
                    "val_mae": best_result["val_metrics"]["mae"],
                    "val_rmse": best_result["val_metrics"]["rmse"],
                    "val_spearman": best_result["val_metrics"]["spearman"],
                    "best_epoch": best_result["best_epoch"],
                }
            )

    trials = pd.DataFrame(all_trials)
    best_configs = pd.DataFrame(best_rows)
    per_bearing = pd.concat(all_per_bearing, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    split_summary, protocol_summary = summarize_final(per_bearing)
    trials.to_csv(out_dir / "tuning_trials.csv", index=False)
    best_configs.to_csv(out_dir / "best_configs_by_split.csv", index=False)
    per_bearing.to_csv(out_dir / "tuned_wavelet_k40_per_bearing_metrics.csv", index=False)
    predictions.to_csv(out_dir / "tuned_wavelet_k40_predictions.csv", index=False)
    split_summary.to_csv(out_dir / "tuned_wavelet_k40_split_summary.csv", index=False)
    protocol_summary.to_csv(out_dir / "tuned_wavelet_k40_protocol_summary.csv", index=False)
    save_figures(split_summary, protocol_summary, fig_dir)
    print(f"\nSaved tuned results to {out_dir}", flush=True)


def save_figures(split_summary: pd.DataFrame, protocol_summary: pd.DataFrame, fig_dir: Path) -> None:
    cross = protocol_summary[protocol_summary["protocol"] == "cross_condition"].copy()
    cross["model"] = pd.Categorical(cross["model"], categories=MODEL_ORDER, ordered=True)
    cross = cross.sort_values("mae_mean")

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=220)
    bars = ax.bar(cross["model"].astype(str), cross["mae_mean"], color=[MODEL_COLORS.get(str(m), "#64748b") for m in cross["model"]])
    ax.set_title("Tuned wavelet-only cross-condition MAE")
    ax.set_ylabel("Bearing-level MAE")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, max(cross["mae_mean"].max() * 1.25, 0.1))
    for bar, value in zip(bars, cross["mae_mean"], strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006, f"{value:.3f}", ha="center", va="bottom", fontsize=9, weight="bold")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(fig_dir / "tuned_wavelet_k40_cross_mae_bar.png", bbox_inches="tight")
    plt.close(fig)

    heat = split_summary[split_summary["protocol"] == "cross_condition"].copy()
    heat["test_condition"] = heat["split_name"].str.extract(r"test_(C\d)")
    pivot = heat.pivot_table(index="model", columns="test_condition", values="mae", aggfunc="mean")
    row_order = [m for m in MODEL_ORDER if m in pivot.index]
    col_order = [c for c in ["C1", "C2", "C3"] if c in pivot.columns]
    pivot = pivot.loc[row_order, col_order]
    fig, ax = plt.subplots(figsize=(5.4, 5.0), dpi=220)
    im = ax.imshow(pivot.to_numpy(), cmap="YlGnBu_r", aspect="auto")
    ax.set_xticks(np.arange(len(col_order)))
    ax.set_xticklabels([f"Test {c}" for c in col_order])
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_order)
    ax.set_title("Tuned per-split cross-condition MAE")
    for i in range(len(row_order)):
        for j in range(len(col_order)):
            val = pivot.iloc[i, j]
            color = "white" if val < np.nanmean(pivot.to_numpy()) else "#0b1f3d"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8.5, weight="bold", color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("MAE")
    fig.tight_layout()
    fig.savefig(fig_dir / "tuned_wavelet_k40_per_split_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    table = cross[["model", "mae_mean", "rmse_mean", "spearman_mean"]].copy()
    table.columns = ["Model", "MAE", "RMSE", "Spearman"]
    for col in ["MAE", "RMSE", "Spearman"]:
        table[col] = table[col].map(lambda x: f"{x:.3f}")
    fig, ax = plt.subplots(figsize=(8.2, 2.8), dpi=220)
    ax.axis("off")
    tbl = ax.table(cellText=table.values, colLabels=table.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.0, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        if r == 0:
            cell.set_facecolor("#e8f0fb")
            cell.set_text_props(weight="bold", color="#0b1f3d")
        elif r == 1:
            cell.set_facecolor("#fff7cc")
            cell.set_text_props(weight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "tuned_wavelet_k40_cross_table.png", bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Validation-tuned wavelet-only K=40 cross-condition experiment.")
    parser.add_argument("--features", default="processed/features_wavelet_only.csv")
    parser.add_argument("--split_dir", default="processed/splits_final")
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--seq_dir", type=Path, default=Path("processed/sequences_wavelet_tuned_k40"))
    parser.add_argument("--out_dir", type=Path, default=Path("results/tables/tuned_wavelet_k40"))
    parser.add_argument("--fig_dir", type=Path, default=Path("results/figures/tuned_wavelet_k40"))
    parser.add_argument("--pred_dir", type=Path, default=Path("results/predictions_tuned_wavelet_k40"))
    parser.add_argument("--ckpt_dir", type=Path, default=Path("results/checkpoints_tuned_wavelet_k40"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--smooth_weight", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge_alphas", nargs="+", type=float, default=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    parser.add_argument("--rebuild_sequences", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
