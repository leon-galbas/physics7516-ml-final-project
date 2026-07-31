"""
AI Assistance Declaration

This script was generated with the assistance of ChatGPT 5.5 Full and was
subsequently reviewed and slightly modified by hand.
"""

import argparse
import copy
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset_builder import DatasetBuilder
from src.training.checkpoint import load_checkpoint, load_model_from_checkpoint
from src.training.evaluation import compute_metrics
from src.utils import get_nested, read_config

logger = logging.getLogger(__name__)


# basic utilities
def experiment_name(exp_dir: str | Path) -> str:
    return Path(exp_dir).name


def load_xy_from_experiment(exp_dir: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    exp_dir = Path(exp_dir)
    X_path = exp_dir / "data" / "X.pt"
    Y_path = exp_dir / "data" / "Y.pt"

    if not X_path.exists():
        raise FileNotFoundError(f"Missing feature tensor: {X_path}")

    if not Y_path.exists():
        raise FileNotFoundError(f"Missing target tensor: {Y_path}")

    X = torch.load(X_path)
    Y = torch.load(Y_path)

    if X.size(0) != Y.size(0):
        raise ValueError(
            f"X and Y have inconsistent sample counts: {X.shape}, {Y.shape}"
        )

    return X, Y


def split_like_training(
    X: torch.Tensor,
    Y: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    test_size = get_nested(config, "training", "test_size", default=0.2)
    seed = get_nested(config, "training", "seed", default=42)

    n = X.size(0)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))

    test_n = int(test_size * n)
    test_idx = perm[:test_n]
    train_idx = perm[test_n:]

    X_train = X[train_idx]
    X_test = X[test_idx]
    Y_train = Y[train_idx]
    Y_test = Y[test_idx]

    return X_train, Y_train, X_test, Y_test


def apply_feature_scaling(
    X: torch.Tensor,
    exp_dir: str | Path,
    config: dict[str, Any],
) -> torch.Tensor:
    if not get_nested(config, "data", "feature_scaling", default=False):
        return X

    scale_path = Path(exp_dir) / "data" / "X_scale_factors.pt"

    if not scale_path.exists():
        raise FileNotFoundError(f"Missing feature scaling factors: {scale_path}")

    means, stds = torch.load(scale_path)
    eps = 1e-8
    return (X - means) / (stds + eps)


def inverse_target_scaling_np(
    Y: np.ndarray,
    exp_dir: str | Path,
    config: dict[str, Any],
) -> np.ndarray:
    if not get_nested(config, "data", "target_scaling", default=False):
        return Y

    scale_path = Path(exp_dir) / "data" / "Y_scale_factors.pt"

    if not scale_path.exists():
        raise FileNotFoundError(f"Missing target scaling factors: {scale_path}")

    means, stds = torch.load(scale_path)
    means_np = means.detach().cpu().numpy()
    stds_np = stds.detach().cpu().numpy()

    return Y * stds_np + means_np


def load_best_model(exp_dir: str | Path, config: dict[str, Any]) -> nn.Module:
    checkpoint_path = Path(exp_dir) / "checkpoints" / "checkpoint_best.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing best checkpoint: {checkpoint_path}")

    model = load_model_from_checkpoint(str(checkpoint_path), config)
    model.eval()
    return model


def predict_numpy(
    model: nn.Module,
    X: torch.Tensor,
    batch_size: int,
    predict_endpoint: bool,
) -> np.ndarray:
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    y_pred_batches = []

    model.eval()
    with torch.no_grad():
        for (X_batch,) in loader:
            prediction = model(X_batch)

            if predict_endpoint:
                if not isinstance(prediction, tuple | list):
                    raise TypeError(
                        "Expected endpoint model to return a tuple/list "
                        "(parameter_prediction, endpoint_prediction)."
                    )
                prediction = torch.cat(prediction, dim=1)
            else:
                if isinstance(prediction, tuple | list):
                    prediction = prediction[0]

            y_pred_batches.append(prediction.detach().cpu().numpy())

    return np.concatenate(y_pred_batches, axis=0)


def target_names_from_config(config: dict[str, Any]) -> list[str]:
    target_names = get_nested(config, "data", "targets")

    if target_names is None:
        raise ValueError("Could not find data.targets in config.")

    return list(target_names)


def metric_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split: str,
    target_names: list[str],
    model_name: str,
    eval_type: str,
    condition_name: str | None = None,
    condition_value: float | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    row = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        split=split,
        target="overall",
        target_index=-1,
    )
    row.update(
        {
            "model": model_name,
            "eval_type": eval_type,
            "condition_name": condition_name,
            "condition_value": condition_value,
        }
    )
    rows.append(row)

    for i, target_name in enumerate(target_names):
        row = compute_metrics(
            y_true=y_true[:, i],
            y_pred=y_pred[:, i],
            split=split,
            target=target_name,
            target_index=i,
        )
        row.update(
            {
                "model": model_name,
                "eval_type": eval_type,
                "condition_name": condition_name,
                "condition_value": condition_value,
            }
        )
        rows.append(row)

    return rows


def save_table(df: pd.DataFrame, output_base: str | Path) -> None:
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    parquet_path = output_base.with_suffix(".parquet")
    csv_path = output_base.with_suffix(".csv")

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    logger.info(f"Saved table to '{parquet_path}' and '{csv_path}'.")


# Loss curves
def plot_loss_curve(exp_dir: str | Path, output_dir: str | Path) -> None:
    exp_dir = Path(exp_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    name = experiment_name(exp_dir)
    checkpoint_dir = exp_dir / "checkpoints"

    checkpoint = load_checkpoint(str(checkpoint_dir), name="checkpoint_final")
    loss_history = checkpoint.get("loss_history", {})

    train_loss = np.asarray(loss_history.get("train_loss", []), dtype=float)
    test_loss = np.asarray(loss_history.get("test_loss", []), dtype=float)

    if len(train_loss) == 0 or len(test_loss) == 0:
        logger.warning(f"No loss history found for '{name}'.")
        return

    epochs = np.arange(1, len(train_loss) + 1)

    loss_df = pd.DataFrame(
        {
            "epoch": epochs,
            "train_loss": train_loss,
            "test_loss": test_loss,
        }
    )
    loss_df.to_csv(output_dir / f"{name}_loss_history.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_loss, label="Train loss")
    plt.plot(epochs, test_loss, label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss curve: {name}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_loss_curve_linear.pdf")
    plt.savefig(output_dir / f"{name}_loss_curve_linear.png", dpi=200)
    plt.close()

    positive_mask = (epochs > 0) & (train_loss > 0) & (test_loss > 0)

    if not np.any(positive_mask):
        logger.warning(
            f"No positive loss values available for log-log plot of '{name}'."
        )
        return

    plt.figure(figsize=(7, 5))
    plt.loglog(
        epochs[positive_mask],
        train_loss[positive_mask],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Train loss",
    )
    plt.loglog(
        epochs[positive_mask],
        test_loss[positive_mask],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Test loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Log-log loss curve: {name}")
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_loss_curve_loglog.pdf")
    plt.savefig(output_dir / f"{name}_loss_curve_loglog.png", dpi=200)
    plt.close()


# baseline train test evaluation
def evaluate_train_test(
    exp_dir: str | Path,
    batch_size: int,
) -> list[dict[str, Any]]:
    exp_dir = Path(exp_dir)
    name = experiment_name(exp_dir)

    config = read_config(str(exp_dir / "config.yaml"))
    target_names = target_names_from_config(config)

    X, Y = load_xy_from_experiment(exp_dir)
    X_train, Y_train, X_test, Y_test = split_like_training(X, Y, config)

    X_train = apply_feature_scaling(X_train, exp_dir, config)
    X_test = apply_feature_scaling(X_test, exp_dir, config)

    model = load_best_model(exp_dir, config)
    predict_endpoint = get_nested(
        config,
        "model",
        "params",
        "predict_endpoint",
        default=False,
    )

    y_train_pred = predict_numpy(
        model=model,
        X=X_train,
        batch_size=batch_size,
        predict_endpoint=predict_endpoint,
    )
    y_test_pred = predict_numpy(
        model=model,
        X=X_test,
        batch_size=batch_size,
        predict_endpoint=predict_endpoint,
    )

    y_train_true = Y_train.detach().cpu().numpy()
    y_test_true = Y_test.detach().cpu().numpy()

    y_train_pred = inverse_target_scaling_np(y_train_pred, exp_dir, config)
    y_test_pred = inverse_target_scaling_np(y_test_pred, exp_dir, config)

    rows = []
    rows.extend(
        metric_rows(
            y_true=y_train_true,
            y_pred=y_train_pred,
            split="train",
            target_names=target_names,
            model_name=name,
            eval_type="train_test",
        )
    )
    rows.extend(
        metric_rows(
            y_true=y_test_true,
            y_pred=y_test_pred,
            split="test",
            target_names=target_names,
            model_name=name,
            eval_type="train_test",
        )
    )

    return rows


# cropping evaluation
def crop_to_fraction(
    X: torch.Tensor,
    frac: float,
    min_points: int = 2,
) -> torch.Tensor:
    if X.ndim != 3:
        raise ValueError(f"Expected time-series input [N, F, T], got {X.shape}.")

    _, _, T = X.shape
    k = max(min_points, int(frac * T))
    return X[:, :, :k]


def evaluate_cropping_sweep(
    exp_dir: str | Path,
    fractions: list[float],
    batch_size: int,
) -> list[dict[str, Any]]:
    exp_dir = Path(exp_dir)
    name = experiment_name(exp_dir)

    config = read_config(str(exp_dir / "config.yaml"))

    if get_nested(config, "model", "name") != "GRU":
        return []

    if get_nested(config, "data", "feature_type") != "timeseries":
        return []

    target_names = target_names_from_config(config)

    X, Y = load_xy_from_experiment(exp_dir)
    _, _, X_test, Y_test = split_like_training(X, Y, config)
    X_test = apply_feature_scaling(X_test, exp_dir, config)

    model = load_best_model(exp_dir, config)
    predict_endpoint = get_nested(
        config,
        "model",
        "params",
        "predict_endpoint",
        default=False,
    )

    y_true = Y_test.detach().cpu().numpy()

    rows = []

    for frac in fractions:
        logger.info(f"Evaluating cropping fraction {frac:.3f} for '{name}'.")

        X_crop = crop_to_fraction(X_test, frac=frac)

        y_pred = predict_numpy(
            model=model,
            X=X_crop,
            batch_size=batch_size,
            predict_endpoint=predict_endpoint,
        )
        y_pred = inverse_target_scaling_np(y_pred, exp_dir, config)

        rows.extend(
            metric_rows(
                y_true=y_true,
                y_pred=y_pred,
                split="test",
                target_names=target_names,
                model_name=name,
                eval_type="cropping",
                condition_name="crop_fraction",
                condition_value=float(frac),
            )
        )

    return rows


# gaussian noise evaluation
def insert_gaussian_noise_transform(
    transforms: list[Any],
    std: float,
    seed: int,
) -> list[Any]:
    new_transforms: list[Any] = []
    inserted = False

    for transform in transforms:
        new_transforms.append(copy.deepcopy(transform))

        is_resample = False
        if isinstance(transform, dict):
            is_resample = "Resample" in transform
        elif isinstance(transform, str):
            is_resample = transform == "Resample"

        if is_resample and not inserted:
            new_transforms.append(
                {
                    "GaussianNoise": {
                        "std": float(std),
                        "seed": int(seed),
                    }
                }
            )
            inserted = True

    if not inserted:
        new_transforms.insert(
            0,
            {
                "GaussianNoise": {
                    "std": float(std),
                    "seed": int(seed),
                }
            },
        )

    return new_transforms


def build_or_load_noisy_dataset(
    train_config: dict[str, Any],
    exp_name: str,
    std: float,
    seed: int,
    cache_dir: str | Path,
    max_samples: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{exp_name}_gaussian_std_{std:g}".replace(".", "p")
    X_path = cache_dir / f"{tag}_X.pt"
    Y_path = cache_dir / f"{tag}_Y.pt"

    if X_path.exists() and Y_path.exists():
        logger.info(f"Loading cached noisy dataset '{tag}'.")
        return torch.load(X_path), torch.load(Y_path)

    noisy_config = copy.deepcopy(train_config)

    noisy_config["data"]["transforms"] = insert_gaussian_noise_transform(
        transforms=noisy_config["data"]["transforms"],
        std=std,
        seed=seed,
    )

    repo_name = get_nested(noisy_config, "simulation", "repo_name")
    n_samples = get_nested(noisy_config, "simulation", "n_samples")

    if max_samples is not None:
        n_samples = min(int(n_samples), int(max_samples))

    builder = DatasetBuilder(repo_name, noisy_config["data"])
    X, Y = builder.build_dataset(n_samples, verbose=True)

    torch.save(X, X_path)
    torch.save(Y, Y_path)

    return X, Y


def evaluate_noise_sweep(
    exp_dir: str | Path,
    noise_levels: list[float],
    batch_size: int,
    cache_dir: str | Path,
    seed: int,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    exp_dir = Path(exp_dir)
    name = experiment_name(exp_dir)

    config = read_config(str(exp_dir / "config.yaml"))

    if get_nested(config, "model", "name") != "GRU":
        return []

    if get_nested(config, "data", "feature_type") != "timeseries":
        return []

    target_names = target_names_from_config(config)

    model = load_best_model(exp_dir, config)
    predict_endpoint = get_nested(
        config,
        "model",
        "params",
        "predict_endpoint",
        default=False,
    )

    rows = []

    for i, std in enumerate(noise_levels):
        logger.info(f"Evaluating Gaussian noise std={std:.6g} for '{name}'.")

        X, Y = build_or_load_noisy_dataset(
            train_config=config,
            exp_name=name,
            std=std,
            seed=seed + i,
            cache_dir=cache_dir,
            max_samples=max_samples,
        )

        _, _, X_test, Y_test = split_like_training(X, Y, config)
        X_test = apply_feature_scaling(X_test, exp_dir, config)

        y_true = Y_test.detach().cpu().numpy()

        y_pred = predict_numpy(
            model=model,
            X=X_test,
            batch_size=batch_size,
            predict_endpoint=predict_endpoint,
        )
        y_pred = inverse_target_scaling_np(y_pred, exp_dir, config)

        rows.extend(
            metric_rows(
                y_true=y_true,
                y_pred=y_pred,
                split="test",
                target_names=target_names,
                model_name=name,
                eval_type="gaussian_noise",
                condition_name="gaussian_std",
                condition_value=float(std),
            )
        )

    return rows


# summary plots
def plot_sweep_metric(
    df: pd.DataFrame,
    eval_type: str,
    condition_name: str,
    metric: str,
    output_path: str | Path,
    title: str,
    target: str = "overall",
    split: str = "test",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = df[
        (df["eval_type"] == eval_type)
        & (df["condition_name"] == condition_name)
        & (df["target"] == target)
        & (df["split"] == split)
    ].copy()

    if len(plot_df) == 0:
        logger.warning(f"No data available for plot '{output_path}'.")
        return

    plt.figure(figsize=(8, 5))

    for model_name, group in plot_df.groupby("model"):
        group = group.sort_values("condition_value")
        plt.plot(
            group["condition_value"],
            group[metric],
            marker="o",
            linewidth=1.5,
            label=model_name,
        )

    plt.xlabel(condition_name.replace("_", " "))
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path.with_suffix(".pdf"))
    plt.savefig(output_path.with_suffix(".png"), dpi=200)
    plt.close()


# main
def main(eval_config_file: str) -> None:
    eval_config = read_config(eval_config_file)

    output_dir = Path(eval_config.get("output_dir", "experiments/evaluation"))
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int(eval_config.get("batch_size", 256))

    experiments = eval_config["experiments"]

    # loss curves
    loss_dir = output_dir / "loss_curves"
    if eval_config.get("make_loss_plots", True):
        for exp_dir in experiments:
            plot_loss_curve(exp_dir, loss_dir)

    # train test metrics for all models
    all_train_test_rows = []
    if eval_config.get("evaluate_train_test", True):
        for exp_dir in experiments:
            logger.info(f"Evaluating train/test metrics for '{exp_dir}'.")
            all_train_test_rows.extend(
                evaluate_train_test(
                    exp_dir=exp_dir,
                    batch_size=batch_size,
                )
            )

        df_train_test = pd.DataFrame(all_train_test_rows)
        save_table(df_train_test, output_dir / "tables" / "train_test_metrics")

    # cropping sweep for GRU models
    all_crop_rows = []
    cropping_config = eval_config.get("cropping_sweep", {})
    if cropping_config.get("enabled", True):
        fractions = [float(x) for x in cropping_config["fractions"]]

        for exp_dir in experiments:
            all_crop_rows.extend(
                evaluate_cropping_sweep(
                    exp_dir=exp_dir,
                    fractions=fractions,
                    batch_size=batch_size,
                )
            )

        df_crop = pd.DataFrame(all_crop_rows)
        save_table(df_crop, output_dir / "tables" / "gru_cropping_metrics")

        plot_sweep_metric(
            df=df_crop,
            eval_type="cropping",
            condition_name="crop_fraction",
            metric=cropping_config.get("metric", "rmse"),
            output_path=output_dir / "plots" / "gru_performance_vs_cropping",
            title="GRU performance vs. observed trajectory fraction",
        )

    # gaussian noise sweep for all GRU models
    all_noise_rows = []
    noise_config = eval_config.get("gaussian_noise_sweep", {})
    if noise_config.get("enabled", True):
        noise_levels = [float(x) for x in noise_config["std_levels"]]
        seed = int(noise_config.get("seed", 123))
        max_samples = noise_config.get("max_samples", None)

        if max_samples is not None:
            max_samples = int(max_samples)

        cache_dir = output_dir / "cached_noisy_datasets"

        for exp_dir in experiments:
            all_noise_rows.extend(
                evaluate_noise_sweep(
                    exp_dir=exp_dir,
                    noise_levels=noise_levels,
                    batch_size=batch_size,
                    cache_dir=cache_dir,
                    seed=seed,
                    max_samples=max_samples,
                )
            )

        df_noise = pd.DataFrame(all_noise_rows)
        save_table(df_noise, output_dir / "tables" / "gru_gaussian_noise_metrics")

        plot_sweep_metric(
            df=df_noise,
            eval_type="gaussian_noise",
            condition_name="gaussian_std",
            metric=noise_config.get("metric", "rmse"),
            output_path=output_dir / "plots" / "gru_performance_vs_gaussian_noise",
            title="GRU performance vs. Gaussian observation noise",
        )

    logger.info("Finished all evaluations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "eval_config",
        help="Evaluation config YAML file.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler()],
    )

    main(args.eval_config)
