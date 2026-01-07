"""
Evaluation Metrics Calculation
Includes all directional and trend similarity metrics
"""

import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)


def calculate_basic_metrics(y_true, y_pred):
    """Calculate basic metrics"""
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'R^2': r2_score(y_true, y_pred)
    }


def calculate_directional_metrics(y_true, y_pred):
    """Calculate directional metrics"""
    if len(y_true) < 2:
        return {'MDA': np.nan, 'DA': np.nan, 'MADL': np.nan}

    true_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)

    # MDA
    mask = true_diff != 0
    mda = np.mean(np.sign(true_diff[mask]) == np.sign(pred_diff[mask])) if np.any(mask) else np.nan

    # DA
    da = np.mean(np.sign(true_diff) == np.sign(pred_diff))

    # MADL
    madl = np.mean(np.abs(np.sign(true_diff) - np.sign(pred_diff))) / 2.0

    return {'MDA': mda, 'DA': da, 'MADL': madl}


def calculate_trend_similarity(y_true, y_pred):
    """Calculate trend similarity metrics"""
    if len(y_true) < 3:
        return {'Corr_Diff': np.nan, 'Cov_Diff': np.nan}

    true_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)

    min_len = min(len(true_diff), len(pred_diff))
    true_diff = true_diff[:min_len]
    pred_diff = pred_diff[:min_len]

    try:
        corr_matrix = np.corrcoef(true_diff, pred_diff)
        corr_diff = corr_matrix[0, 1]
        cov_diff = np.cov(true_diff, pred_diff)[0, 1]
    except:
        corr_diff = np.nan
        cov_diff = np.nan

    return {'Corr_Diff': corr_diff, 'Cov_Diff': cov_diff}