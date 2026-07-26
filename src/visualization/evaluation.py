import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def predict(
    model: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute model predictions.

    Assumes X, Y, and model are on CPU.
    """

    logger.info(f"Computing predictions for dataset of shape {X.shape}.")

    model.eval()
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    y_true_batches = []
    y_pred_batches = []

    with torch.no_grad():
        for X_batch, Y_batch in loader:
            prediction = model(X_batch)

            if isinstance(prediction, tuple | list):
                prediction = prediction[0]

            y_true_batches.append(Y_batch.numpy())
            y_pred_batches.append(prediction.numpy())

    y_true = np.concatenate(y_true_batches, axis=0)
    y_pred = np.concatenate(y_pred_batches, axis=0)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch between targets and predictions: "
            f"{y_true.shape} vs {y_pred.shape}."
        )

    return y_true, y_pred


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split: str,
    target: str,
    target_index: int,
) -> dict:
    """
    Compute regression metrics for one target or for all targets flattened together.
    """

    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if y_true.size == 0:
        raise ValueError(
            f"No finite values found for split='{split}', target='{target}'."
        )

    residuals = y_pred - y_true
    abs_errors = np.abs(residuals)

    mse = float(np.mean(residuals**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_errors))

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    y_var = float(np.var(y_true))
    residual_var = float(np.var(residuals))
    explained_variance = float(1.0 - residual_var / y_var) if y_var > 0 else np.nan

    nonzero_mask = np.abs(y_true) > 1e-12
    if np.any(nonzero_mask):
        mape = float(
            np.mean(
                np.abs(
                    (y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask]
                )
            )
            * 100.0
        )
    else:
        mape = np.nan

    smape_denominator = np.abs(y_true) + np.abs(y_pred)
    smape_mask = smape_denominator > 1e-12
    if np.any(smape_mask):
        smape = float(
            np.mean(2.0 * abs_errors[smape_mask] / smape_denominator[smape_mask])
            * 100.0
        )
    else:
        smape = np.nan

    if y_true.size > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson_correlation = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson_correlation = np.nan

    return {
        "split": split,
        "target": target,
        "target_index": target_index,
        "n_values": int(y_true.size),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "median_absolute_error": float(np.median(abs_errors)),
        "max_absolute_error": float(np.max(abs_errors)),
        "r2": r2,
        "explained_variance": explained_variance,
        "mape_percent": mape,
        "smape_percent": smape,
        "bias_mean_error": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "pearson_correlation": pearson_correlation,
    }


def build_metrics_dataframe(
    y_train_true: np.ndarray,
    y_train_pred: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    target_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build one dataframe containing overall and per-target regression metrics.
    """

    n_outputs = y_train_true.shape[1]

    if target_names is None:
        target_names = [f"target_{i}" for i in range(n_outputs)]

    if len(target_names) != n_outputs:
        raise ValueError(f"Expected {n_outputs} target names, got {len(target_names)}.")

    rows = []

    datasets = {
        "train": (y_train_true, y_train_pred),
        "test": (y_test_true, y_test_pred),
    }

    for split, (y_true, y_pred) in datasets.items():
        rows.append(
            compute_metrics(
                y_true=y_true,
                y_pred=y_pred,
                split=split,
                target="overall",
                target_index=-1,
            )
        )

        for i, target_name in enumerate(target_names):
            rows.append(
                compute_metrics(
                    y_true=y_true[:, i],
                    y_pred=y_pred[:, i],
                    split=split,
                    target=target_name,
                    target_index=i,
                )
            )

    return pd.DataFrame(rows)


def save_metrics_dataframe(metrics: pd.DataFrame, output_dir: str | Path) -> Path:
    """
    Save metrics dataframe as Parquet.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.parquet"
    metrics.to_parquet(metrics_path, index=False)

    logger.info(f"Saved metrics to '{metrics_path}'.")
    return metrics_path


def sample_points(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_points: int = 20_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample points for scatter plots.
    """

    if y_true.size <= max_points:
        return y_true, y_pred

    rng = np.random.default_rng(seed)
    indices = rng.choice(y_true.size, size=max_points, replace=False)

    return y_true[indices], y_pred[indices]


def plot_predicted_vs_true(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
    title: str,
    max_points: int = 20_000,
) -> None:
    """
    Plot predicted values against true values.
    """

    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    y_true, y_pred = sample_points(y_true, y_pred, max_points=max_points)

    min_value = min(np.min(y_true), np.min(y_pred))
    max_value = max(np.max(y_true), np.max(y_pred))

    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, s=8, alpha=0.35)
    plt.plot([min_value, max_value], [min_value, max_value], "r--", label="Ideal")
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_residuals_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
    title: str,
    max_points: int = 20_000,
) -> None:
    """
    Plot residuals $y_{pred} - y_{true}$ against predicted values.
    """

    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    y_true, y_pred = sample_points(y_true, y_pred, max_points=max_points)
    residuals = y_pred - y_true

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, s=8, alpha=0.35)
    plt.axhline(0.0, color="red", linestyle="--")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_residual_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
    title: str,
    bins: int = 80,
) -> None:
    """
    Plot a histogram of residuals.
    """

    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    residuals = y_pred[mask] - y_true[mask]

    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=bins, alpha=0.8)
    plt.axvline(0.0, color="red", linestyle="--")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_absolute_error_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
    title: str,
    bins: int = 80,
) -> None:
    """
    Plot a histogram of absolute errors.
    """

    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    absolute_errors = np.abs(y_pred[mask] - y_true[mask])

    plt.figure(figsize=(8, 6))
    plt.hist(absolute_errors, bins=bins, alpha=0.8)
    plt.xlabel("Absolute error")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_per_target_predicted_vs_true(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
    title: str,
    target_names: list[str],
    max_targets: int = 16,
    max_points: int = 5_000,
) -> None:
    """
    Plot predicted versus true values separately for each target.
    """

    n_targets = min(y_true.shape[1], max_targets)
    n_cols = min(4, n_targets)
    n_rows = math.ceil(n_targets / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 5 * n_rows),
        squeeze=False,
    )
    fig.suptitle(title)

    for i in range(n_rows * n_cols):
        ax = axes[i // n_cols][i % n_cols]

        if i >= n_targets:
            ax.axis("off")
            continue

        yt = y_true[:, i]
        yp = y_pred[:, i]

        mask = np.isfinite(yt) & np.isfinite(yp)
        yt = yt[mask]
        yp = yp[mask]

        yt, yp = sample_points(yt, yp, max_points=max_points, seed=i)

        min_value = min(np.min(yt), np.min(yp))
        max_value = max(np.max(yt), np.max(yp))

        ax.scatter(yt, yp, s=8, alpha=0.35)
        ax.plot([min_value, max_value], [min_value, max_value], "r--")
        ax.set_title(target_names[i])
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_per_target_residual_boxplot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
    title: str,
    target_names: list[str],
    max_targets: int = 40,
) -> None:
    """
    Plot residual distributions for each target as a boxplot.
    """

    errors = []
    labels = []

    for i in range(min(y_true.shape[1], max_targets)):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        mask = np.isfinite(yt) & np.isfinite(yp)
        target_errors = yp[mask] - yt[mask]

        if target_errors.size > 0:
            errors.append(target_errors)
            labels.append(target_names[i])

    if len(errors) == 0:
        logger.warning(f"No finite residuals available for plot '{output_path}'.")
        return

    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(errors)), 6))

    ax.boxplot(errors, showfliers=False)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.axhline(0.0, color="red", linestyle="--")
    ax.set_xlabel("Target")
    ax.set_ylabel("Residual")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_per_target_metric(
    metrics: pd.DataFrame,
    split: str,
    metric: str,
    output_path: str | Path,
    title: str,
    max_targets: int = 40,
) -> None:
    """
    Plot one metric per target.
    """

    target_metrics = metrics[
        (metrics["split"] == split) & (metrics["target"] != "overall")
    ].copy()

    target_metrics = target_metrics.sort_values("target_index").head(max_targets)

    plt.figure(figsize=(max(8, 0.45 * len(target_metrics)), 6))
    plt.bar(target_metrics["target"], target_metrics[metric])
    plt.xlabel("Target")
    plt.ylabel(metric)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_all_evaluation_graphs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: pd.DataFrame,
    split: str,
    output_dir: str | Path,
    target_names: list[str],
) -> None:
    """
    Create all evaluation plots for one split.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving {split} plots to '{output_dir}'.")

    plot_predicted_vs_true(
        y_true,
        y_pred,
        output_dir / "predicted_vs_true.png",
        title=f"{split.capitalize()}: Predicted vs True",
    )

    plot_residuals_vs_predicted(
        y_true,
        y_pred,
        output_dir / "residuals_vs_predicted.png",
        title=f"{split.capitalize()}: Residuals vs Predicted",
    )

    plot_residual_histogram(
        y_true,
        y_pred,
        output_dir / "residual_histogram.png",
        title=f"{split.capitalize()}: Residual Histogram",
    )

    plot_absolute_error_histogram(
        y_true,
        y_pred,
        output_dir / "absolute_error_histogram.png",
        title=f"{split.capitalize()}: Absolute Error Histogram",
    )

    plot_per_target_predicted_vs_true(
        y_true,
        y_pred,
        output_dir / "per_target_predicted_vs_true.png",
        title=f"{split.capitalize()}: Per-target Predicted vs True",
        target_names=target_names,
    )

    plot_per_target_residual_boxplot(
        y_true,
        y_pred,
        output_dir / "per_target_residual_boxplot.png",
        title=f"{split.capitalize()}: Per-target Residual Boxplot",
        target_names=target_names,
    )

    for metric in ["rmse", "mae", "r2"]:
        plot_per_target_metric(
            metrics,
            split=split,
            metric=metric,
            output_path=output_dir / f"per_target_{metric}.png",
            title=f"{split.capitalize()}: Per-target {metric}",
        )


def main(
    model: nn.Module,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    output_dir: str | Path,
    batch_size: int = 256,
    target_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Evaluate a trained regression model.

    Saves:
    - metrics as `metrics.parquet`
    - plots in `plots/train/` and `plots/test/`

    Returns:
    - metrics dataframe
    """

    logger.info("Starting model evaluation.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if X_train.size(0) != Y_train.size(0):
        raise ValueError(
            f"The number of train samples in X and Y do not match "
            f"({X_train.size(0)} vs {Y_train.size(0)})."
        )

    if X_test.size(0) != Y_test.size(0):
        raise ValueError(
            f"The number of test samples in X and Y do not match "
            f"({X_test.size(0)} vs {Y_test.size(0)})."
        )

    y_train_true, y_train_pred = predict(model, X_train, Y_train, batch_size=batch_size)
    y_test_true, y_test_pred = predict(model, X_test, Y_test, batch_size=batch_size)

    if y_train_true.shape[1] != y_test_true.shape[1]:
        raise ValueError(
            f"Train and test targets have different output dimensions "
            f"({y_train_true.shape[1]} vs {y_test_true.shape[1]})."
        )

    n_outputs = y_train_true.shape[1]

    if target_names is None:
        target_names = [f"target_{i}" for i in range(n_outputs)]

    if len(target_names) != n_outputs:
        raise ValueError(f"Expected {n_outputs} target names, got {len(target_names)}.")

    metrics = build_metrics_dataframe(
        y_train_true=y_train_true,
        y_train_pred=y_train_pred,
        y_test_true=y_test_true,
        y_test_pred=y_test_pred,
        target_names=target_names,
    )

    save_metrics_dataframe(metrics, output_dir)

    plot_all_evaluation_graphs(
        y_true=y_train_true,
        y_pred=y_train_pred,
        metrics=metrics,
        split="train",
        output_dir=output_dir / "plots" / "train",
        target_names=target_names,
    )

    plot_all_evaluation_graphs(
        y_true=y_test_true,
        y_pred=y_test_pred,
        metrics=metrics,
        split="test",
        output_dir=output_dir / "plots" / "test",
        target_names=target_names,
    )

    logger.info("Finished model evaluation.")

    return metrics
