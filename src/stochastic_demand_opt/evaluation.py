import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def pinball_loss(y_true, y_pred, q):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    err = y_true - y_pred
    return np.mean(np.maximum(q * err, (q - 1) * err))

def interval_coverage(y_true, lower, upper):
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    return np.mean((y_true >= lower) & (y_true <= upper))

def interval_width(lower, upper):
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    return np.mean(upper - lower)

def summarize_predictions(df_pred, label="set"):
    y = df_pred["y_true"].values
    p10 = df_pred["y_pred_q_0.1"].values
    p50 = df_pred["y_pred_q_0.5"].values
    p90 = df_pred["y_pred_q_0.9"].values
    naive = df_pred["y_pred_naive"].values

    mae = mean_absolute_error(y, p50)
    rmse = np.sqrt(mean_squared_error(y, p50))
    cov = interval_coverage(y, p10, p90)
    wid = interval_width(p10, p90)

    mae_naive = mean_absolute_error(y, naive)
    rmse_naive = np.sqrt(mean_squared_error(y, naive))

    pb10 = pinball_loss(y, p10, 0.10)
    pb50 = pinball_loss(y, p50, 0.50)
    pb90 = pinball_loss(y, p90, 0.90)

    return pd.Series({
        "set": label,
        "mae_q50": mae,
        "rmse_q50": rmse,
        "pinball_q10": pb10,
        "pinball_q50": pb50,
        "pinball_q90": pb90,
        "avg_pinball_10_50_90": (pb10 + pb50 + pb90) / 3.0,
        "coverage_10_90": cov,
        "avg_width_10_90": wid,
        "mae_naive": mae_naive,
        "rmse_naive": rmse_naive,
        "mae_improvement_vs_naive_pct": 100 * (mae_naive - mae) / mae_naive if mae_naive > 0 else np.nan,
        "rmse_improvement_vs_naive_pct": 100 * (rmse_naive - rmse) / rmse_naive if rmse_naive > 0 else np.nan,
    })
