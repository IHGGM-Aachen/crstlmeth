"""
crstlmeth/core/stats.py
--------------------
statistical utilities for one-sample tests, FDR correction,
and normal-parameter estimation from quantiles
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, t
from statsmodels.stats.multitest import multipletests


def one_sample_z_test(
    sample_levels: np.ndarray,
    target_levels: np.ndarray,
    *,
    axis: int = 0,
    fdr_alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    one-sample z-test: target vs cohort (two-sided), BH-FDR over all tests

    sample_levels : array with cohort values (e.g. shape (n_refs, n_regions))
    target_levels : array with target values (broadcastable against cohort mean/sd)
    axis          : axis along which cohort mean/sd are computed
    returns       : (z_scores, pvals, p_adj, flags)
    """
    mean_ref = np.nanmean(sample_levels, axis=axis)
    std_ref = np.nanstd(sample_levels, axis=axis, ddof=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        z_scores = np.divide(
            target_levels - mean_ref,
            std_ref,
            out=np.zeros_like(target_levels, dtype=float),
            where=np.isfinite(std_ref) & (std_ref > 0),
        )

    pvals = 2.0 * (1.0 - norm.cdf(np.abs(z_scores)))

    # BH-FDR across all elements
    flat = pvals.reshape(-1)
    _, p_adj_flat, _, _ = multipletests(flat, alpha=fdr_alpha, method="fdr_bh")
    p_adj = p_adj_flat.reshape(pvals.shape)
    flags = p_adj < fdr_alpha

    return z_scores, pvals, p_adj, flags


def one_sample_t_test(
    sample_mat: np.ndarray,
    target_vec: np.ndarray,
    *,
    axis: int = 0,
    fdr_alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    one-sample t-test: target vs cohort (two-sided), BH-FDR over all tests
    """
    n = sample_mat.shape[axis]
    mean_ref = np.nanmean(sample_mat, axis=axis)
    std_ref = np.nanstd(sample_mat, axis=axis, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        se = std_ref / np.sqrt(n)
        t_stats = np.divide(
            target_vec - mean_ref,
            se,
            out=np.zeros_like(target_vec, dtype=float),
            where=np.isfinite(se) & (se > 0),
        )
    pvals = 2.0 * (1.0 - t.cdf(np.abs(t_stats), df=max(n - 1, 1)))
    flat = pvals.reshape(-1)
    _, p_adj_flat, _, _ = multipletests(flat, alpha=fdr_alpha, method="fdr_bh")
    p_adj = p_adj_flat.reshape(pvals.shape)
    flags = p_adj < fdr_alpha
    return t_stats, pvals, p_adj, flags


def approx_normal_params_from_quantiles(
    q25: np.ndarray,
    q50: np.ndarray,
    q75: np.ndarray,
    q10: np.ndarray | None = None,
    q90: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    estimate (mu, sigma) from reference quantiles for an approx. normal model.
    primary sigma = (q75 - q25) / 1.349; fallback = (q90 - q10) / (2 * 1.281)
    """
    mu = np.asarray(q50, dtype=float)
    iqr = np.asarray(q75, dtype=float) - np.asarray(q25, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = iqr / 1.349

    if (q10 is not None) and (q90 is not None):
        span = np.asarray(q90, dtype=float) - np.asarray(q10, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_alt = span / (2.0 * 1.281)
        # prefer alt where finite
        sigma = np.where(np.isfinite(sigma_alt), sigma_alt, sigma)

    # guard non-positive/NaN sigma
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nan)
    return mu, sigma
