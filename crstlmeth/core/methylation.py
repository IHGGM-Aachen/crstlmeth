"""
crstlmeth/core/methylation.py

routines for extracting and analyzing targeted methylation levels
from bedmethyl files over a defined set of intervals
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .parsers import get_region_stats
from .stats import one_sample_z_test


class Methylation:
    """
    encapsulate methylation-level computations from .bedmethyl.gz inputs
    """

    # ────────────────────────────────────────────────────────────────────
    # pooled levels
    # ────────────────────────────────────────────────────────────────────
    @staticmethod
    def get_levels(
        bedmethyl_files: list[str],
        intervals: list[tuple[str, int, int]],
    ) -> np.ndarray:
        """
        compute methylation level (nmod / nvalid_cov) for each sample and interval

        groups files by naive sample id (prefix before first underscore) and pools
        all parts (hap1/hap2/ungrouped) of each sample.
        """

        def _sid(p: str) -> str:
            import pathlib as _pl

            return _pl.Path(p).name.split("_")[0]

        groups: dict[str, list[str]] = {}
        for p in bedmethyl_files:
            groups.setdefault(_sid(p), []).append(p)

        n_samps = len(groups)
        n_inter = len(intervals)
        levels = np.zeros((n_samps, n_inter), dtype=float)

        for i, (_sid, paths) in enumerate(sorted(groups.items())):
            for j, (chrom, start, end) in enumerate(intervals):
                tot_m = 0
                tot_v = 0
                for path in paths:
                    m, v = get_region_stats(path, chrom, start, end)
                    tot_m += m
                    tot_v += v
                levels[i, j] = (tot_m / tot_v) if tot_v > 0 else np.nan

        return levels

    # ────────────────────────────────────────────────────────────────────
    # hap-aware helpers
    # ────────────────────────────────────────────────────────────────────
    @staticmethod
    def get_levels_by_haplotype(
        hap_paths: dict[str, str],  # keys "1","2","ungrouped" (any subset)
        intervals: list[tuple[str, int, int]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        returns three arrays of shape (n_intervals,):
            hap1_levels, hap2_levels, overall_levels
        """
        m1 = np.zeros(len(intervals))
        v1 = np.zeros(len(intervals))
        m2 = np.zeros(len(intervals))
        v2 = np.zeros(len(intervals))
        mu = np.zeros(len(intervals))
        vu = np.zeros(len(intervals))

        for j, (c, s, e) in enumerate(intervals):
            if "1" in hap_paths:
                m1[j], v1[j] = get_region_stats(hap_paths["1"], c, s, e)
            if "2" in hap_paths:
                m2[j], v2[j] = get_region_stats(hap_paths["2"], c, s, e)
            if "ungrouped" in hap_paths:
                mu[j], vu[j] = get_region_stats(hap_paths["ungrouped"], c, s, e)

        hap1 = np.divide(
            m1, v1, out=np.full_like(m1, np.nan, dtype=float), where=v1 > 0
        )
        hap2 = np.divide(
            m2, v2, out=np.full_like(m2, np.nan, dtype=float), where=v2 > 0
        )
        overall = np.divide(
            m1 + m2 + mu,
            v1 + v2 + vu,
            out=np.full_like(mu, np.nan, dtype=float),
            where=(v1 + v2 + vu) > 0,
        )
        return hap1, hap2, overall

    @staticmethod
    def build_haplotype_reference_matrix(
        cohort_files_by_sample: dict[str, list[str]],
        intervals: list[tuple[str, int, int]],
    ) -> np.ndarray:
        """
        build a reference matrix where each ROW is a hap observation from the cohort
        (all hap1 rows, hap2 rows, and optionally pooled rows if only one hap present).
        """
        rows = []
        for _sid, paths in sorted(cohort_files_by_sample.items()):
            # classify parts per sample
            h: dict[str, str] = {}
            from pathlib import Path as _P

            for p in paths:
                name = _P(p).name
                if "_1." in name and "bedmethyl" in name:
                    h["1"] = p
                elif "_2." in name and "bedmethyl" in name:
                    h["2"] = p
                elif "_ungrouped." in name and "bedmethyl" in name:
                    h["ungrouped"] = p

            if not h:
                # pool arbitrary pieces if provided
                lv = Methylation.get_levels(paths, intervals)[0]
                rows.append(lv)
                continue

            # compute hap1/hap2/overall per sample
            hap1, hap2, overall = Methylation.get_levels_by_haplotype(
                h, intervals
            )
            if not np.all(np.isnan(hap1)):
                rows.append(hap1)
            if not np.all(np.isnan(hap2)):
                rows.append(hap2)
            if not np.all(np.isnan(overall)):
                rows.append(overall)

        return (
            np.vstack(rows)
            if rows
            else np.zeros((0, len(intervals)), dtype=float)
        )

    # ────────────────────────────────────────────────────────────────────
    # stats
    # ────────────────────────────────────────────────────────────────────
    @staticmethod
    def get_confidence_intervals(
        bedmethyl_files: list[str],
        intervals: list[tuple[str, int, int]],
        z_score: float = 1.96,
    ) -> np.ndarray:
        """
        normal-approximate CI for each methylation level:
            p ± z * sqrt(p*(1−p)/n)
        """
        n_samps = len(bedmethyl_files)
        n_inter = len(intervals)
        cis = np.zeros((n_samps, n_inter, 2), dtype=float)

        for i, path in enumerate(bedmethyl_files):
            for j, (chrom, start, end) in enumerate(intervals):
                mod, valid = get_region_stats(path, chrom, start, end)
                if valid > 0:
                    p = mod / valid
                    m = z_score * np.sqrt(p * (1 - p) / valid)
                    cis[i, j] = (p - m, p + m)
                else:
                    cis[i, j] = (np.nan, np.nan)

        return cis

    @staticmethod
    def get_deviations(
        sample_levels: np.ndarray,
        target_levels: np.ndarray,
        fdr_alpha: float = 0.05,
    ) -> np.ndarray:
        """
        one-sample z-test vs cohort + BH-FDR
        returns boolean array shape (n_targets, n_intervals)
        """
        _, _, _, flags = one_sample_z_test(
            sample_levels, target_levels, fdr_alpha=fdr_alpha
        )
        return flags

    @staticmethod
    def filter_empty_regions(
        levels_list: list[np.ndarray],
        intervals: list[tuple[str, int, int]],
        region_names: list[str],
    ):
        """
        remove regions where all samples have NaN or 0 in all arrays
        """
        empties = []
        for lv in levels_list:
            empties.append(np.all(np.isnan(lv) | (lv == 0), axis=0))
        mask = np.all(empties, axis=0)

        for k in range(len(levels_list)):
            levels_list[k] = levels_list[k][:, ~mask]
        intervals[:] = [iv for i, iv in enumerate(intervals) if not mask[i]]
        region_names[:] = [
            rn for i, rn in enumerate(region_names) if not mask[i]
        ]

    @staticmethod
    def assess_phasing_quality(
        hap_paths: dict[str, str],
        intervals: list[tuple[str, int, int]],
        thresh: float = 0.40,
    ) -> dict:
        """
        fraction of ungrouped reads per interval (as a phasing quality heuristic)
        """
        n = len(intervals)
        vu = np.zeros(n)
        v1 = np.zeros(n)
        v2 = np.zeros(n)

        for j, (c, s, e) in enumerate(intervals):
            if "ungrouped" in hap_paths:
                _, vu[j] = get_region_stats(hap_paths["ungrouped"], c, s, e)
            if "1" in hap_paths:
                _, v1[j] = get_region_stats(hap_paths["1"], c, s, e)
            if "2" in hap_paths:
                _, v2[j] = get_region_stats(hap_paths["2"], c, s, e)

        denom = v1 + v2 + vu
        frac = np.divide(vu, denom, out=np.full(n, np.nan), where=denom > 0)
        mask = np.where(np.isnan(frac), False, frac >= thresh)

        return {
            "frac_ungrouped": frac,
            "flag_mask": mask,
            "threshold": float(thresh),
            "n_flagged": int(mask.sum()),
            "n_regions": int(n),
        }

    # ────────────────────────────────────────────────────────────────────
    # hap-aware matching
    # ────────────────────────────────────────────────────────────────────
    @dataclass(frozen=True)
    class HapMatchResult:
        clear: bool
        mapping: Dict[str, str]  # {"hap1": "1"|"2", "hap2": "1"|"2"}
        scores: Dict[
            str, float
        ]  # "h1_vs_ref1", "h1_vs_ref2", "h2_vs_ref1", "h2_vs_ref2"
        n_used_min: int  # min #regions used across pairwise scores
        mode: str  # "aggregated" | "full"

    # ---- helpers to build hap-quantiles with name OR coordinate alignment ----
    @staticmethod
    def _align_by_region_then_coords(
        df: pd.DataFrame,
        regions: List[str],
        intervals: Optional[List[tuple[str, int, int]]] = None,
    ) -> pd.DataFrame:
        """
        Try to align by 'region' names; if too sparse and coordinates exist,
        fall back to coordinate-based alignment and then restore the requested order.
        """
        # First try: by region name
        if "region" in df.columns:
            df1 = df.drop_duplicates(subset="region", keep="first").set_index(
                "region"
            )
            df1 = df1.reindex(regions)
        else:
            df1 = df.copy()
            df1.index = regions

        # If this is good enough, use it
        if df1.notna().sum(axis=1).mean() >= 1 or intervals is None:
            return df1

        # Fallback: coordinate alignment
        if not {"chrom", "start", "end"}.issubset(df.columns):
            return df1  # nothing better we can do

        # Build desired order by intervals
        order = pd.DataFrame(
            {
                "chrom": [c for (c, s, e) in intervals],
                "start": [s for (c, s, e) in intervals],
                "end": [e for (c, s, e) in intervals],
                "region": regions,
            }
        )
        # Drop duplicates on coord on both sides, left-join to preserve order
        left = order.drop_duplicates(
            subset=["chrom", "start", "end"], keep="first"
        )
        right = df.drop_duplicates(
            subset=["chrom", "start", "end"], keep="first"
        )
        merged = left.merge(
            right,
            on=["chrom", "start", "end"],
            how="left",
            suffixes=("", "_ref"),
        )
        merged = merged.set_index("region")
        return merged

    @staticmethod
    def _median_abs_z_from_quantiles(
        q25: np.ndarray,
        q50: np.ndarray,
        q75: np.ndarray,
        target: np.ndarray,
        *,
        sd: np.ndarray | None = None,
    ) -> Tuple[float, int]:
        """
        robust score = median( | (target - q50) / sigma | )
        with sigma estimated from IQR (sigma_iqr = (q75-q25)/1.349),
        and falling back to 'sd' where IQR≈0 or missing.

        returns (score, n_used)
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            iqr = q75 - q25
            sigma_iqr = iqr / 1.349
            sigma = np.where(
                np.isfinite(sigma_iqr) & (sigma_iqr > 0), sigma_iqr, np.nan
            )
            if sd is not None:
                sigma = np.where(
                    (~np.isfinite(sigma)) | (sigma <= 0), sd, sigma
                )

            valid = (
                np.isfinite(target)
                & np.isfinite(q50)
                & np.isfinite(sigma)
                & (sigma > 0)
            )
            if not np.any(valid):
                return float("inf"), 0

            zabs = np.abs((target[valid] - q50[valid]) / sigma[valid])
            return float(np.median(zabs)), int(valid.sum())

    @staticmethod
    def _aggregated_quantiles_for_hap(
        df_meth: pd.DataFrame,
        regions: List[str],
        hap_key: str,
        intervals: Optional[List[tuple[str, int, int]]] = None,
    ) -> pd.DataFrame:
        """
        quantile frame (index=regions) for aggregated reference for a given hap_key.
        Uses coordinate fallback if region names don't match.
        Includes 'sd' column as dispersion fallback.
        """
        df = df_meth
        if "section" in df.columns:
            df = df[df["section"].astype(str) == "meth"].copy()
        if "hap_key" in df.columns:
            df = df[df["hap_key"].astype(str) == str(hap_key)].copy()
        else:
            df = df.iloc[0:0].copy()
        if df.empty:
            return pd.DataFrame(
                index=regions, columns=["q25", "q50", "q75", "sd"], dtype=float
            )

        df = Methylation._align_by_region_then_coords(df, regions, intervals)

        def pick(col: str, fb: str | None = None) -> np.ndarray:
            if col in df.columns:
                return df[col].to_numpy(dtype=float)
            if fb and fb in df.columns:
                return df[fb].to_numpy(dtype=float)
            return np.full(len(df), np.nan)

        out = pd.DataFrame(
            {
                "q25": pick("meth_q25"),
                "q50": pick("meth_median", "meth_q50"),
                "q75": pick("meth_q75"),
                "sd": pick("meth_sd"),
            },
            index=df.index,
        )
        return out.reindex(regions)

    @staticmethod
    def _full_quantiles_for_hap(
        df_full: pd.DataFrame,
        regions: List[str],
        hap_key: str,
        intervals: Optional[List[tuple[str, int, int]]] = None,
    ) -> pd.DataFrame:
        """
        cohort quantiles per region from full reference rows for a hap.
        Uses coordinate fallback if region names don't match (requires intervals).
        """
        df = df_full[df_full.get("hap", "").astype(str) == str(hap_key)].copy()
        if df.empty:
            return pd.DataFrame(
                index=regions, columns=["q25", "q50", "q75", "sd"], dtype=float
            )

        # pivot by region name first
        piv = df.pivot_table(
            index="sample_id", columns="region", values="meth", aggfunc="first"
        )

        # If name alignment poor and intervals available, rebuild frame by coords:
        if intervals is not None and len(set(piv.columns) & set(regions)) < max(
            5, int(0.2 * len(regions))
        ):
            # compute per-region mean across cohort, attach coords from original df, then align via coords
            mean_by_region = piv.mean(axis=0)
            coords = df.drop_duplicates(subset=["region"]).set_index("region")[
                ["chrom", "start", "end"]
            ]
            tmp = pd.DataFrame(
                {"region": mean_by_region.index, "q50": mean_by_region.values}
            ).set_index("region")
            tmp = tmp.join(coords, how="left").reset_index()

            order = pd.DataFrame(
                {
                    "chrom": [c for (c, s, e) in intervals],
                    "start": [s for (c, s, e) in intervals],
                    "end": [e for (c, s, e) in intervals],
                    "region": regions,
                }
            )
            merged = order.merge(
                tmp, on=["chrom", "start", "end"], how="left"
            ).set_index("region")
            # fake q25/q75 via nanstd (symmetric around q50) as a fallback
            arr = piv.reindex(
                columns=list(tmp["region"].dropna().unique()), fill_value=np.nan
            ).to_numpy(dtype=float)
            sd = np.nanstd(arr, axis=0, ddof=0)
            q25 = merged["q50"].to_numpy(dtype=float) - 0.6745 * np.nan_to_num(
                sd, nan=np.nan
            )
            q75 = merged["q50"].to_numpy(dtype=float) + 0.6745 * np.nan_to_num(
                sd, nan=np.nan
            )
            out = pd.DataFrame(
                {
                    "q25": q25,
                    "q50": merged["q50"].to_numpy(dtype=float),
                    "q75": q75,
                    "sd": sd[: len(q25)],
                },
                index=merged.index,
            )
            return out.reindex(regions)

        # normal case: name alignment fine → quantiles from array
        arr = piv.reindex(columns=regions, fill_value=np.nan).to_numpy(
            dtype=float
        )
        q25 = np.nanpercentile(arr, 25, axis=0)
        q50 = np.nanpercentile(arr, 50, axis=0)
        q75 = np.nanpercentile(arr, 75, axis=0)
        sd = np.nanstd(arr, axis=0, ddof=0)
        return pd.DataFrame(
            {"q25": q25, "q50": q50, "q75": q75, "sd": sd}, index=regions
        )

    @staticmethod
    def hap_quantiles_for_reference(
        ref_df: pd.DataFrame,
        mode: str,
        regions: List[str],
        hap_key: str,
        intervals: Optional[List[tuple[str, int, int]]] = None,
    ) -> pd.DataFrame:
        """
        Unified accessor for hap-specific quantiles for both reference modes.
        Always returns q25,q50,q75,sd (sd may be NaN).
        """
        m = (mode or "aggregated").lower()
        if m == "aggregated":
            return Methylation._aggregated_quantiles_for_hap(
                ref_df, regions, hap_key, intervals
            )
        if m == "full":
            return Methylation._full_quantiles_for_hap(
                ref_df, regions, hap_key, intervals
            )
        return pd.DataFrame(
            index=regions, columns=["q25", "q50", "q75", "sd"], dtype=float
        )

    @staticmethod
    def auto_map_target_haps(
        ref_df: pd.DataFrame,
        mode: str,
        regions: List[str],
        target_h1: np.ndarray,
        target_h2: np.ndarray,
        *,
        intervals: Optional[List[tuple[str, int, int]]] = None,
        min_regions: int = 10,
        ratio_ambiguous: float = 1.05,  # relaxed default
        delta_ambiguous: float = 0.08,  # relaxed default
    ) -> "Methylation.HapMatchResult":
        """
        Decide mapping between target hap1/2 and reference hap 1/2 using robust |z|-median distance.

        (1) Build hap-specific quantiles (aligned by name; fallback to coords).
        (2) Score each target hap vs ref-hap using median |(target - q50)/sigma|, sigma from
            IQR/1.349 with fallback to 'sd' where IQR≈0.
        (3) Ambiguous if (for either hap):
              max(score)/min(score) < ratio_ambiguous  OR  |Δscore| < delta_ambiguous
            OR insufficient aligned regions (< min_regions).
        """
        q1 = Methylation.hap_quantiles_for_reference(
            ref_df, mode, regions, "1", intervals=intervals
        )
        q2 = Methylation.hap_quantiles_for_reference(
            ref_df, mode, regions, "2", intervals=intervals
        )

        # If reference lacks hap-aware info, bail ambiguous
        if (
            q1[["q25", "q50", "q75"]].isna().all().all()
            or q2[["q25", "q50", "q75"]].isna().all().all()
        ):
            return Methylation.HapMatchResult(
                False,
                {"hap1": "1", "hap2": "2"},
                {},
                0,
                (mode or "aggregated").lower(),
            )

        # Ensure targets align to regions length
        t1 = np.asarray(target_h1, dtype=float)
        t2 = np.asarray(target_h2, dtype=float)
        L = min(
            len(regions), t1.shape[0], t2.shape[0], q1.shape[0], q2.shape[0]
        )
        t1 = t1[:L]
        t2 = t2[:L]
        q1 = q1.iloc[:L, :]
        q2 = q2.iloc[:L, :]

        s_h1_r1, n1 = Methylation._median_abs_z_from_quantiles(
            q1["q25"].values,
            q1["q50"].values,
            q1["q75"].values,
            t1,
            sd=q1.get("sd", pd.Series(np.nan, index=q1.index)).to_numpy(),
        )
        s_h1_r2, n2 = Methylation._median_abs_z_from_quantiles(
            q2["q25"].values,
            q2["q50"].values,
            q2["q75"].values,
            t1,
            sd=q2.get("sd", pd.Series(np.nan, index=q2.index)).to_numpy(),
        )
        s_h2_r1, n3 = Methylation._median_abs_z_from_quantiles(
            q1["q25"].values,
            q1["q50"].values,
            q1["q75"].values,
            t2,
            sd=q1.get("sd", pd.Series(np.nan, index=q1.index)).to_numpy(),
        )
        s_h2_r2, n4 = Methylation._median_abs_z_from_quantiles(
            q2["q25"].values,
            q2["q50"].values,
            q2["q75"].values,
            t2,
            sd=q2.get("sd", pd.Series(np.nan, index=q2.index)).to_numpy(),
        )
        n_used_min = min(n1 + n2, n3 + n4)

        if n_used_min < int(min_regions):
            return Methylation.HapMatchResult(
                False,
                {"hap1": "1", "hap2": "2"},
                {
                    "h1_vs_ref1": s_h1_r1,
                    "h1_vs_ref2": s_h1_r2,
                    "h2_vs_ref1": s_h2_r1,
                    "h2_vs_ref2": s_h2_r2,
                },
                n_used_min,
                (mode or "aggregated").lower(),
            )

        def _clear(a: float, b: float) -> bool:
            mn, mx = min(a, b), max(a, b)
            return (mx / (mn if mn > 0 else 1.0) >= ratio_ambiguous) or (
                abs(a - b) >= delta_ambiguous
            )

        clear_h1 = _clear(s_h1_r1, s_h1_r2)
        clear_h2 = _clear(s_h2_r1, s_h2_r2)
        mapping = {
            "hap1": ("1" if s_h1_r1 <= s_h1_r2 else "2"),
            "hap2": ("1" if s_h2_r1 <= s_h2_r2 else "2"),
        }

        return Methylation.HapMatchResult(
            clear=bool(clear_h1 and clear_h2),
            mapping=mapping,
            scores={
                "h1_vs_ref1": s_h1_r1,
                "h1_vs_ref2": s_h1_r2,
                "h2_vs_ref1": s_h2_r1,
                "h2_vs_ref2": s_h2_r2,
            },
            n_used_min=n_used_min,
            mode=(mode or "aggregated").lower(),
        )
