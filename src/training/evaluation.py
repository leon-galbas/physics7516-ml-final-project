import numpy as np


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
