import argparse
import logging
from datetime import datetime
from os import makedirs, path
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.config import LOGS_DIR
from src.data.dataset_builder import DatasetBuilder
from src.simulation.generator import TrajectoryGenerator
from src.training.checkpoint import (
    load_model_from_checkpoint,
)
from src.utils import get_nested, read_config

logger = logging.getLogger(__name__)


LOSS_FUNCS = {
    "MSE": nn.MSELoss,
    "MAE": nn.L1Loss,
}


def main(
    config_file: str,
    checkpoint: str,
    X_scale_factors: str = "",
    Y_scale_factors: str = "",
) -> None:
    logger.info(f"Starting the experiment specified in '{config_file}'.")

    # initialize directories
    directory = path.dirname(config_file)
    data_dir = path.join(directory, "data")
    eval_dir = path.join(directory, "evaluation")
    makedirs(data_dir, exist_ok=True)
    makedirs(eval_dir, exist_ok=True)

    # read config
    config = read_config(config_file)

    # generate trajectories
    sim_config = config["simulation"]
    n_samples = sim_config["n_samples"]
    generator = TrajectoryGenerator(**sim_config)
    generator.generate_to_repo(sim_config["repo_name"], n_samples, verbose=True)

    # load data
    features_path = path.join(data_dir, "X.pt")
    targets_path = path.join(data_dir, "Y.pt")
    if path.exists(features_path) and path.exists(targets_path):
        logger.info("The processed dataset already exists. Loading data...")
        X = torch.load(features_path)
        logger.info(f"Loaded feature dataset of shape {X.shape}.")
        Y = torch.load(targets_path)
        logger.info(f"Loaded target dataset of shape {Y.shape}.")
    else:
        repo_name = get_nested(config, "simulation", "repo_name")
        builder = DatasetBuilder(repo_name, config["data"])  # pyright: ignore[reportArgumentType]
        X, Y = builder.build_dataset(n_samples, verbose=True)
        torch.save(X, features_path)
        torch.save(Y, targets_path)
    if X.size(0) != Y.size(0):
        raise ValueError(
            f"The number of samples in X and Y do not match ({X.size(0)} vs {Y.size(0)})."
        )

    # scale data
    feature_type = get_nested(config, "data", "feature_type")
    feature_scaling = get_nested(config, "data", "feature_scaling")
    target_scaling = get_nested(config, "data", "target_scaling")
    if feature_scaling:
        if feature_type not in ["timeseries", "scalar"]:
            raise ValueError(
                "Feature type in data config must be either 'timeseries' or 'scalar', "
                f"got '{feature_type}'."
            )
        if X_scale_factors:
            eps = 1e-8
            means, stds = torch.load(X_scale_factors)
            X = (X - means) / (stds + eps)
        else:
            raise ValueError(
                "No scaling factors provided for feature scaling. "
                "Use argument 'X_scale_factors'."
            )
    if target_scaling:
        if Y_scale_factors:
            eps = 1e-8
            means, stds = torch.load(Y_scale_factors)
            Y = (Y - means) / (stds + eps)
        else:
            raise ValueError(
                "No scaling factors provided for target scaling. "
                "Use argument 'Y_scale_factors'."
            )

    # create batches
    data = TensorDataset(X, Y)
    batch_size = get_nested(config, "evaluation", "batch_size", default=256)
    loader = DataLoader(
        data,
        batch_size=batch_size,  # pyright: ignore[reportArgumentType]
        shuffle=False,
    )

    # load checkpoint
    model = load_model_from_checkpoint(checkpoint, config)
    model.eval()

    # setup evaluation
    cropping = get_nested(config, "evaluation", "cropping", default=1.0)
    predict_endpoint = get_nested(
        config, "model", "params", "predict_endpoint", default=False
    )

    # run evaluation
    logger.info("Running evaluation...")
    y_true_batches = []
    y_pred_batches = []

    with torch.no_grad():
        for X_batch, Y_batch in tqdm(loader):
            if feature_type == "timeseries" and cropping < 1.0:  # pyright: ignore[reportOperatorIssue]
                X_batch = crop_to_fraction(X_batch, cropping)  # pyright: ignore[reportArgumentType]

            if predict_endpoint:
                prediction = torch.cat(model(X_batch), dim=1)
            else:
                prediction = model(X_batch)

            y_true_batches.append(Y_batch.numpy())
            y_pred_batches.append(prediction.numpy())

    y_true = np.concatenate(y_true_batches, axis=0)
    y_pred = np.concatenate(y_pred_batches, axis=0)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch between targets and predictions.")

    # save final checkpoint
    logger.info("Computing metrics...")
    target_names = get_nested(config, "data", "targets")
    rows = []
    rows.append(
        compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            target="overall",
            target_index=-1,
        )
    )

    for i, target_name in enumerate(target_names):  # pyright: ignore[reportArgumentType]
        rows.append(
            compute_metrics(
                y_true=y_true[:, i],
                y_pred=y_pred[:, i],
                target=target_name,
                target_index=i,
            )
        )

    metrics = pd.DataFrame(rows)
    metrics.to_parquet(path.join(eval_dir, "metrics.pq"))
    logger.info(f"Evaluation metrics:\n{metrics}")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target: str,
    target_index: int,
) -> dict:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if y_true.size == 0:
        raise ValueError(f"No finite values found for target='{target}'.")

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


def random_crop_batch_timeseries(
    X_batch: torch.Tensor,
    min_frac: float = 0.2,
    max_frac: float = 1.0,
    min_points: int = 2,
) -> torch.Tensor:
    if X_batch.ndim != 3:
        raise ValueError(
            "Expected timeseries input of shape [batch_size, features, timepoints], "
            f"got {X_batch.shape}."
        )

    _, _, T = X_batch.shape

    min_len = max(min_points, int(min_frac * T))
    max_len = max(min_len, int(max_frac * T))

    k = torch.randint(
        low=min_len,
        high=max_len + 1,
        size=(1,),
    ).item()

    return X_batch[:, :, :k]


def crop_to_fraction(
    X_batch: torch.Tensor,
    frac: float,
    min_points: int = 2,
) -> torch.Tensor:
    if X_batch.ndim != 3:
        raise ValueError(
            "Expected timeseries input of shape [batch_size, features, timepoints], "
            f"got {X_batch.shape}."
        )

    _, _, T = X_batch.shape
    k = max(min_points, int(frac * T))

    return X_batch[:, :, :k]


if __name__ == "__main__":
    # logger setup
    script_name = Path(__file__).stem
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    logfile = path.join(LOGS_DIR, f"{timestamp}_{script_name}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logfile),
        ],
    )

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Name of the training configuration.")
    parser.add_argument("checkpoint_file", help="Model checkpoint to evaluate.")
    parser.add_argument(
        "-x",
        "--x-scale-factors",
        type=str,
        default="",
        help="Input file for feature scaling factors.",
    )
    parser.add_argument(
        "-y",
        "--y-scale-factors",
        type=str,
        default="",
        help="Input file for target scaling factors.",
    )
    args = parser.parse_args()

    main(
        args.config_file,
        args.checkpoint_file,
        X_scale_factors=args.x_scale_factors,
        Y_scale_factors=args.y_scale_factors,
    )
