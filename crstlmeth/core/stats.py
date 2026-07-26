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
    one-sample z-test: target vs cohort (two-sided), BH-FDR over valid tests only

    sample_levels : array with cohort values (e.g. shape (n_refs, n_regions))
    target_levels : array with target values (broadcastable against cohort mean/sd)
    axis          : axis along which cohort mean/sd are computed
    returns       : (z_scores, pvals, p_adj, flags)
    """
    sample_levels = np.asarray(sample_levels, dtype=float)
    target_levels = np.asarray(target_levels, dtype=float)

    mean_ref = np.nanmean(sample_levels, axis=axis)
    std_ref = np.nanstd(sample_levels, axis=axis, ddof=0)

    valid = (
        np.isfinite(target_levels)
        & np.isfinite(mean_ref)
        & np.isfinite(std_ref)
        & (std_ref > 0)
    )

    z_scores = np.full_like(target_levels, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(
            target_levels - mean_ref,
            std_ref,
            out=z_scores,
            where=valid,
        )

    pvals = np.full_like(z_scores, np.nan, dtype=float)
    pvals[valid] = 2.0 * (1.0 - norm.cdf(np.abs(z_scores[valid])))

    flat = pvals.reshape(-1)
    valid_flat = np.isfinite(flat)

    p_adj_flat = np.full_like(flat, np.nan, dtype=float)
    if valid_flat.any():
        _, adj, _, _ = multipletests(
            flat[valid_flat],
            alpha=fdr_alpha,
            method="fdr_bh",
        )
        p_adj_flat[valid_flat] = adj

    p_adj = p_adj_flat.reshape(pvals.shape)
    flags = np.isfinite(p_adj) & (p_adj < fdr_alpha)

    return z_scores, pvals, p_adj, flags


def one_sample_t_test(
    sample_mat: np.ndarray,
    target_vec: np.ndarray,
    *,
    axis: int = 0,
    fdr_alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    one-sample t-test: target vs cohort (two-sided), BH-FDR over valid tests only
    """
    sample_mat = np.asarray(sample_mat, dtype=float)
    target_vec = np.asarray(target_vec, dtype=float)

    n_eff = np.sum(np.isfinite(sample_mat), axis=axis)
    mean_ref = np.nanmean(sample_mat, axis=axis)
    std_ref = np.nanstd(sample_mat, axis=axis, ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        se = std_ref / np.sqrt(n_eff)

    valid = (
        np.isfinite(target_vec)
        & np.isfinite(mean_ref)
        & np.isfinite(se)
        & (se > 0)
        & (n_eff >= 2)
    )

    t_stats = np.full_like(target_vec, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(
            target_vec - mean_ref,
            se,
            out=t_stats,
            where=valid,
        )

    pvals = np.full_like(t_stats, np.nan, dtype=float)
    df = np.maximum(n_eff - 1, 1)
    df_b = np.broadcast_to(np.asarray(df, dtype=float), t_stats.shape)
    pvals[valid] = 2.0 * (1.0 - t.cdf(np.abs(t_stats[valid]), df=df_b[valid]))

    flat = pvals.reshape(-1)
    valid_flat = np.isfinite(flat)

    p_adj_flat = np.full_like(flat, np.nan, dtype=float)
    if valid_flat.any():
        _, adj, _, _ = multipletests(
            flat[valid_flat],
            alpha=fdr_alpha,
            method="fdr_bh",
        )
        p_adj_flat[valid_flat] = adj

    p_adj = p_adj_flat.reshape(pvals.shape)
    flags = np.isfinite(p_adj) & (p_adj < fdr_alpha)

    return t_stats, pvals, p_adj, flags


def approx_normal_params_from_quantiles(
    q25: np.ndarray,
    q50: np.ndarray,
    q75: np.ndarray,
    q10: np.ndarray | None = None,
    q90: np.ndarray | None = None,
    q05: np.ndarray | None = None,
    q95: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    estimate (mu, sigma) from reference quantiles for an approx. normal model.

    primary sigma = (q75 - q25) / 1.349

    fallbacks:
      - (q90 - q10) / (2 * 1.2815515655)
      - (q95 - q05) / (2 * 1.6448536270)
    """
    mu = np.asarray(q50, dtype=float)
    iqr = np.asarray(q75, dtype=float) - np.asarray(q25, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = iqr / 1.349

    if (q10 is not None) and (q90 is not None):
        span = np.asarray(q90, dtype=float) - np.asarray(q10, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_alt = span / (2.0 * 1.2815515655446004)
        sigma = np.where(
            np.isfinite(sigma_alt) & (sigma_alt > 0),
            sigma_alt,
            sigma,
        )

    elif (q05 is not None) and (q95 is not None):
        span = np.asarray(q95, dtype=float) - np.asarray(q05, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_alt = span / (2.0 * 1.6448536269514722)
        sigma = np.where(
            np.isfinite(sigma_alt) & (sigma_alt > 0),
            sigma_alt,
            sigma,
        )

    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nan)
    return mu, sigma
