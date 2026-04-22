#!/usr/bin/env python3
"""
sfp_satellite.py
================

Spectral Feature Polynomial (SFP) framework for satellite multispectral imagery.

Reference paper: Lotfi et al. (2026), "Interpretable Machine Learning-Derived
Spectral Indices for Vegetation Monitoring", Machine Learning with Applications.

This script discovers a single interpretable spectral index for any binary
"target vegetation vs. everything else" classification task on a satellite
multispectral dataset. Because satellite scenes differ in illumination,
atmospheric attenuation, and view geometry, the search is restricted to the
three illumination-INVARIANT core families (ND, ND3, NCurv). The resulting
index is a closed-form algebraic function of reflectance bands that is
exactly invariant to global multiplicative scaling.

--------------------------------------------------------------------------
How to prepare your data
--------------------------------------------------------------------------

Put your data in a single CSV (or multiple CSVs, one per acquisition date).
The CSV must have:

  1. One column per spectral band, containing reflectance values (any scale --
     the normalized-difference families cancel out multiplicative scaling).

  2. One categorical "label" column. Pick ONE value in this column to be your
     "target vegetation" (e.g., 'Kochia', 'Cotton', 'SugarBeet', 'Wetland', ...).
     Every row whose label equals that target is treated as the positive class
     (y = 1); every other row is treated as background / "everything else"
     (y = 0). The framework is binary: target vs. not-target.

  3. (Optional) a year / date / acquisition-id column, to enable
     leave-one-year-out cross-validation.

  4. (Optional) a GeoJSON-polygon geometry column, to enable spatial and
     spatio-temporal block cross-validation.

--------------------------------------------------------------------------
Typical invocation
--------------------------------------------------------------------------

Minimum (one CSV, no spatial or temporal structure -- just random 10-fold):

    python sfp_satellite.py \\
        --csvs mydata.csv \\
        --label-col LandCover --target Wetland \\
        --bands B2,B3,B4,B5,B6,B7,B8,B11,B12

Full (paper's Kochia setup -- multi-year with polygons):

    python sfp_satellite.py \\
        --csvs year2022.csv year2023.csv year2024.csv \\
        --label-col Type --target Kochia \\
        --bands b1,b2,b3,b4,b5,b6,b7,b9,b10 \\
        --year-col Year --geom-col .geo

When --year-col is provided, year-held-out CV is added.
When --geom-col is provided, 3x3 spatial and spatio-temporal block CV are added.
Without either, only Random 10-fold is run.

License: MIT
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from sfp_core import (
        build_core_basis,
        discover_index,
        eval_single_feature_svm,
        load_and_concat_csvs,
        polygon_centroid,
        assign_spatial_blocks,
        random_kfold,
        year_held_out_folds,
        block_held_out_folds,
        spatiotemporal_folds,
        EPS,
    )
except ImportError:
    print(
        "ERROR: sfp_core.py must be in the same directory as sfp_satellite.py.",
        file=sys.stderr,
    )
    raise


# =============================================================================
# Optional: established vegetation-index baselines
# =============================================================================


def compute_baselines(
    d: pd.DataFrame, band_map: Dict[str, str]
) -> Dict[str, np.ndarray]:
    """Compute established baseline indices when the user has mapped band roles.

    `band_map` maps a role ('Blue', 'Green', 'Red', 'RE1', 'NIR', 'SWIR1') to
    the CSV column name that contains that physical band. If any required role
    is missing for a particular baseline, that baseline is silently skipped.
    """
    def col(role: str) -> Optional[np.ndarray]:
        name = band_map.get(role)
        if name is None or name not in d.columns:
            return None
        return d[name].values

    Blue, Green, Red = col("Blue"), col("Green"), col("Red")
    RE1, NIR = col("RE1"), col("NIR")
    baselines: Dict[str, np.ndarray] = {}
    if NIR is not None and Red is not None:
        baselines["NDVI"] = (NIR - Red) / (NIR + Red + EPS)
        baselines["SAVI"] = 1.5 * (NIR - Red) / (NIR + Red + 0.5 + EPS)
        if Blue is not None:
            baselines["EVI"] = (
                2.5 * (NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1 + EPS)
            )
    if NIR is not None and Green is not None:
        baselines["GNDVI"] = (NIR - Green) / (NIR + Green + EPS)
    if NIR is not None and RE1 is not None:
        baselines["NDRE"] = (NIR - RE1) / (NIR + RE1 + EPS)
        baselines["CIre"] = NIR / (RE1 + EPS) - 1
    return baselines


# =============================================================================
# Pipeline
# =============================================================================


def run_sfp_satellite(
    csv_paths: List[str],
    band_cols: List[str],
    label_col: str,
    target_class: str,
    year_col: Optional[str] = None,
    geom_col: Optional[str] = None,
    C: float = 0.5,
    baseline_role_map: Optional[Dict[str, str]] = None,
    verbose: bool = True,
) -> Dict:
    """Run the SFP framework on satellite multispectral data.

    Parameters
    ----------
    csv_paths : list of str
        One or more CSV file paths. Rows are concatenated.
    band_cols : list of str
        Column names holding reflectance values.
    label_col : str
        Categorical column whose values define the positive / negative classes.
    target_class : str
        Value in `label_col` that marks the positive ("target vegetation") class.
        All other rows are negative ("everything else").
    year_col : str or None
        Optional; enables leave-one-year-out cross-validation.
    geom_col : str or None
        Optional (GeoJSON Polygon strings); enables 3x3 spatial-block and
        spatio-temporal cross-validation.
    C : float
        SVM regularization parameter.
    baseline_role_map : dict or None
        Optional mapping {'Blue','Green','Red','RE1','NIR','SWIR1'} -> CSV column,
        to compute NDVI/NDRE/CIre/SAVI/EVI/GNDVI baselines alongside the SFP winner.
    """
    if verbose:
        print("=" * 72)
        print("SFP satellite pipeline (Lotfi et al. 2026, MLWA)")
        print("  Basis: illumination-invariant core families (ND, ND3, NCurv)")
        print("=" * 72)

    d = load_and_concat_csvs(
        csv_paths=csv_paths,
        band_cols=band_cols,
        label_col=label_col,
        target_class=target_class,
        year_col=year_col,
        geom_col=geom_col,
    )
    X = d[band_cols].values
    y = (d[label_col].astype(str) == target_class).astype(int).values

    if verbose:
        print(f"\nData: {len(d):,} samples from {len(csv_paths)} file(s)")
        print(f"  Target '{target_class}' (positive): {int(y.sum()):,}")
        print(f"  Everything else (negative):         {int((1 - y).sum()):,}")
        print(f"  Bands used ({len(band_cols)}): {band_cols}")

    years = d[year_col].values if year_col else None
    block = None
    if geom_col is not None:
        cc = [polygon_centroid(g) for g in d[geom_col]]
        lons = np.array([c[0] for c in cc])
        lats = np.array([c[1] for c in cc])
        block = assign_spatial_blocks(lons, lats, grid=3)

    # Enable whichever CV strategies the data supports
    cv_strategies: Dict[str, list] = {
        "Random 10-fold": random_kfold(y, n_splits=10, seed=42)
    }
    if years is not None:
        folds = year_held_out_folds(years, y)
        if folds:
            cv_strategies["Year-held-out"] = folds
        elif verbose:
            print("  (skipping Year-held-out: need 2+ distinct years with mixed classes)")
    if block is not None:
        folds = block_held_out_folds(block, y)
        if folds:
            cv_strategies["Spatial block"] = folds
        elif verbose:
            print("  (skipping Spatial block: insufficient valid blocks)")
    if (
        years is not None
        and block is not None
        and len(np.unique(years)) > 1
        and len(np.unique(block)) > 1
    ):
        folds = spatiotemporal_folds(years, block, y)
        if folds:
            cv_strategies["Spatio-temporal"] = folds
        elif verbose:
            print("  (skipping Spatio-temporal: no valid year x block cell)")

    if verbose:
        print("\nCV strategies enabled:")
        for name, folds in cv_strategies.items():
            print(f"  {name}: {len(folds)} folds")

    if verbose:
        print("\nBuilding invariant-core basis...")
    basis, basis_names = build_core_basis(X, band_cols)
    if verbose:
        print(f"  {len(basis_names):,} basis features "
              f"(degree-2 expansion yields ~{len(basis_names) ** 2 // 2:,} candidates)")

    all_folds = []
    for folds in cv_strategies.values():
        all_folds.extend(folds)

    if verbose:
        print(f"\nRunning per-fold selection across {len(all_folds)} total folds...")
    winner_name, winner_recipe, counter, winner_feature = discover_index(
        basis, basis_names, y, all_folds, verbose=verbose
    )

    # Per-strategy accuracy using the consensus winner
    if verbose:
        print("\nAccuracy of consensus index:")
        print(f"  {'Strategy':<20s} {'Folds':>6s} {'Mean':>8s} {'Median':>8s} {'Min':>8s}")
    strategy_results: Dict = {}
    aggregated: List[float] = []
    for sname, folds in cv_strategies.items():
        accs = [eval_single_feature_svm(winner_feature, y, tr, te, C=C) for tr, te in folds]
        strategy_results[sname] = {
            "folds": len(accs),
            "mean": float(np.mean(accs)),
            "median": float(np.median(accs)),
            "min": float(np.min(accs)),
            "accs": accs,
        }
        aggregated.extend(accs)
        if verbose:
            print(f"  {sname:<20s} {len(accs):>6d} "
                  f"{np.mean(accs) * 100:>7.2f}% "
                  f"{np.median(accs) * 100:>7.2f}% "
                  f"{np.min(accs) * 100:>7.2f}%")
    if verbose and len(cv_strategies) > 1:
        print(f"  {'Overall':<20s} {len(aggregated):>6d} "
              f"{np.mean(aggregated) * 100:>7.2f}% "
              f"{np.median(aggregated) * 100:>7.2f}% "
              f"{np.min(aggregated) * 100:>7.2f}%")

    # Optional baselines
    baseline_results: Dict = {}
    if baseline_role_map:
        baselines = compute_baselines(d, baseline_role_map)
        if baselines:
            baselines["Discovered"] = winner_feature
            if verbose:
                print("\nBaseline comparison (mean accuracy per strategy):")
                header = f"  {'Index':<12s}"
                for sname in cv_strategies:
                    header += f" {sname[:12]:>12s}"
                print(header)
            for bname, feat in baselines.items():
                row = {}
                for sname, folds in cv_strategies.items():
                    accs = [
                        eval_single_feature_svm(feat, y, tr, te, C=C)
                        for tr, te in folds
                    ]
                    row[sname] = float(np.mean(accs))
                baseline_results[bname] = row
                if verbose:
                    line = f"  {bname:<12s}"
                    for sname in cv_strategies:
                        line += f" {row[sname] * 100:>11.2f}%"
                    print(line)

    return {
        "winner_name": winner_name,
        "winner_recipe": winner_recipe,
        "selection_counter": dict(counter),
        "strategy_results": strategy_results,
        "baseline_results": baseline_results,
    }


# =============================================================================
# CLI
# =============================================================================


def _parse_comma_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_role_map(s: Optional[str]) -> Optional[Dict[str, str]]:
    """Parse 'Blue=b1,Red=b3,NIR=b7' -> {'Blue':'b1','Red':'b3','NIR':'b7'}."""
    if not s:
        return None
    out: Dict[str, str] = {}
    for pair in s.split(","):
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main():
    ap = argparse.ArgumentParser(
        description=(
            "SFP framework for satellite imagery (Lotfi et al. 2026, MLWA). "
            "Discovers one interpretable spectral index for a binary 'target "
            "vegetation vs. everything else' task using only scale-invariant "
            "normalized-difference families."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example (paper's Kochia reproduction):\n"
            "  python sfp_satellite.py \\\n"
            "      --csvs IndexNorm_to_points_2022_cleaned.csv "
            "IndexNorm_to_points_2023_cleaned.csv IndexNorm_to_points_2024_cleaned.csv \\\n"
            "      --label-col Type --target Kochia \\\n"
            "      --bands b1,b2,b3,b4,b5,b6,b7,b9,b10 \\\n"
            "      --year-col Year --geom-col .geo \\\n"
            "      --baseline-role-map 'Blue=b1,Green=b2,Red=b3,RE1=b4,NIR=b7,SWIR1=b9'"
        ),
    )
    ap.add_argument("--csvs", nargs="+", required=True,
                    help="One or more CSV file paths. If multiple, rows concatenated.")
    ap.add_argument("--bands", required=True,
                    help="Comma-separated band column names (e.g., b1,b2,b3,b4,b5).")
    ap.add_argument("--label-col", required=True,
                    help="Name of the categorical label column.")
    ap.add_argument("--target", required=True,
                    help="Value in --label-col that marks the target vegetation. "
                         "All other rows are treated as 'everything else'.")
    ap.add_argument("--year-col", default=None,
                    help="(Optional) year/date column; enables year-held-out CV.")
    ap.add_argument("--geom-col", default=None,
                    help="(Optional) GeoJSON Polygon column; enables spatial and "
                         "spatio-temporal block CV.")
    ap.add_argument("--C", type=float, default=0.5,
                    help="SVM regularization parameter (default: 0.5).")
    ap.add_argument("--baseline-role-map", default=None,
                    help="(Optional) comma-separated Role=Column pairs to enable "
                         "baseline comparison (NDVI/NDRE/CIre/SAVI/EVI/GNDVI). "
                         "Roles: Blue,Green,Red,RE1,NIR,SWIR1. Example: "
                         "'Blue=b1,Green=b2,Red=b3,RE1=b4,NIR=b7'.")
    args = ap.parse_args()

    run_sfp_satellite(
        csv_paths=args.csvs,
        band_cols=_parse_comma_list(args.bands),
        label_col=args.label_col,
        target_class=args.target,
        year_col=args.year_col,
        geom_col=args.geom_col,
        C=args.C,
        baseline_role_map=_parse_role_map(args.baseline_role_map),
    )


if __name__ == "__main__":
    main()
