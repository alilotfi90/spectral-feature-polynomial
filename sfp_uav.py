#!/usr/bin/env python3
"""
sfp_uav.py
==========

Spectral Feature Polynomial (SFP) framework for calibrated UAV / drone imagery.

Reference paper: Lotfi et al. (2026), "Interpretable Machine Learning-Derived
Spectral Indices for Vegetation Monitoring", Machine Learning with Applications.

This script discovers a single interpretable spectral index for any binary
"target vegetation vs. everything else" classification task on calibrated
UAV / drone multispectral imagery. Because drone imagery is typically
radiometrically calibrated to surface reflectance via a pre-flight reference
panel, absolute reflectance magnitudes carry biological signal, so the search
is done over the FULL extended basis of 13 feature families (including the
scale-sensitive HAbs family).

The script supports a *multi-task* workflow in which the same framework is
applied independently to several related CSVs that represent different
acquisition windows / growth stages / flight dates. Each task produces its
own deployable index.

--------------------------------------------------------------------------
How to prepare your data
--------------------------------------------------------------------------

One CSV per task (a "task" is a single acquisition date, or a growth stage,
or any grouping in which the spectral signature of the target is roughly
stable). Each CSV must have:

  1. One column per spectral band, containing calibrated reflectance values.

  2. One categorical "label" column. Pick ONE value in this column to be your
     "target vegetation" (e.g., 'Canopy', 'Corn', 'Barley', 'Pinus', ...).
     Every row whose label equals that target is treated as the positive class
     (y = 1); every other row is treated as background / "everything else"
     (y = 0). The framework is binary: target vs. not-target.

  3. One "block" column. Two-fold spatial-block cross-validation requires
     every row to be labelled as belonging to one of two spatially separated
     blocks (conventionally `block=1` or `block=2`). To create blocks, draw a
     horizontal or vertical line across your field in your labelling tool and
     assign all polygons on one side of the line to block 1 and all on the
     other side to block 2. This disjoint separation is what prevents spatial
     autocorrelation leakage, which would otherwise inflate accuracy estimates
     on high-resolution UAV imagery.

--------------------------------------------------------------------------
Typical invocation
--------------------------------------------------------------------------

Single-task (one flight / one date):

    python sfp_uav.py \\
        --csvs myflight.csv \\
        --label-col Type --target Canopy \\
        --bands b1,b2,b3,b4,b5 \\
        --block-col block \\
        --Cs 0.1

Multi-task (one CSV per growth stage, each with its own tuned C):

    python sfp_uav.py \\
        --csvs stage1.csv stage2.csv stage3.csv \\
        --label-col Type --target Canopy \\
        --bands b1,b2,b3,b4,b5 \\
        --block-col block \\
        --Cs 0.01,0.1,0.25

Paper-reference mode (reproduces the manuscript's published WCI_1/WCI_2/WCI_3
formulas on the MicaSense five-band wheat data; only meaningful for that
specific sensor and those specific tasks):

    python sfp_uav.py --paper-indices \\
        --csvs T1_blocked.csv T2_blocked.csv T3_blocked.csv \\
        --label-col Type --target Canopy \\
        --bands b1,b2,b3,b4,b5 \\
        --block-col block \\
        --Cs 0.01,0.1,0.25

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
        build_extended_basis,
        discover_index,
        eval_single_feature_svm,
        load_and_concat_csvs,
        two_block_folds,
        EPS,
    )
except ImportError:
    print(
        "ERROR: sfp_core.py must be in the same directory as sfp_uav.py.",
        file=sys.stderr,
    )
    raise


# =============================================================================
# Paper's reference WCI formulas (MicaSense RedEdge 5-band layout)
#
# These are hard-coded to the paper's exact sensor and band ordering:
#   b1 = Blue (475 nm), b2 = Green (560 nm), b3 = Red (668 nm),
#   b4 = Red-edge (717 nm), b5 = NIR (840 nm)
#
# Using --paper-indices only makes sense when your CSV follows this layout.
# =============================================================================


def paper_wci_1(X: np.ndarray) -> np.ndarray:
    """Paper's early-season WCI_1: -ND3(+b1,-b3,+b4) * SR2((b2+b3)/(b1+b5))."""
    b1, b2, b3, b4, b5 = (X[:, i] for i in range(5))
    nd3 = (b1 - b3 + b4) / (b1 + b3 + b4 + EPS)
    sr2 = (b2 + b3) / (b1 + b5 + EPS)
    return -nd3 * sr2


def paper_wci_2(X: np.ndarray) -> np.ndarray:
    """Paper's mid-season WCI_2: -SR3((b1+b2+b4)/b3) * SR3((b1+b3+b4)/b5)."""
    b1, b2, b3, b4, b5 = (X[:, i] for i in range(5))
    sr3a = (b1 + b2 + b4) / (b3 + EPS)
    sr3b = (b1 + b3 + b4) / (b5 + EPS)
    return -sr3a * sr3b


def paper_wci_3(X: np.ndarray) -> np.ndarray:
    """Paper's late-season WCI_3: ND(b1,b3) * HAbs(b3)."""
    b1, b2, b3, b4, b5 = (X[:, i] for i in range(5))
    nd = (b1 - b3) / (b1 + b3 + EPS)
    max_other = np.maximum.reduce([b1, b2, b4, b5])
    habs = np.tanh(b3 - max_other)
    return nd * habs


PAPER_WCI = {1: paper_wci_1, 2: paper_wci_2, 3: paper_wci_3}


# =============================================================================
# Optional: established baseline vegetation indices (user-specified band roles)
# =============================================================================


def compute_baselines(
    d: pd.DataFrame, band_map: Dict[str, str]
) -> Dict[str, np.ndarray]:
    """Compute established UAV baselines when the user has mapped band roles.

    Supports NDVI, NDRE, CIre, GNDVI, NDWI, SAVI, EVI2, GRRI, SR.
    Roles required: Blue, Green, Red, RE, NIR.
    """
    def col(role: str) -> Optional[np.ndarray]:
        name = band_map.get(role)
        if name is None or name not in d.columns:
            return None
        return d[name].values

    B, G, R = col("Blue"), col("Green"), col("Red")
    RE, NIR = col("RE"), col("NIR")
    bl: Dict[str, np.ndarray] = {}
    if NIR is not None and R is not None:
        bl["NDVI"] = (NIR - R) / (NIR + R + EPS)
        bl["SAVI"] = 1.5 * (NIR - R) / (NIR + R + 0.5 + EPS)
        bl["EVI2"] = 2.5 * (NIR - R) / (NIR + 2.4 * R + 1 + EPS)
        bl["SR"] = NIR / (R + EPS)
    if NIR is not None and G is not None:
        bl["GNDVI"] = (NIR - G) / (NIR + G + EPS)
    if NIR is not None and RE is not None:
        bl["NDRE"] = (NIR - RE) / (NIR + RE + EPS)
        bl["CIre"] = NIR / (RE + EPS) - 1
    if NIR is not None and G is not None:
        bl["NDWI"] = (G - NIR) / (G + NIR + EPS)
    if G is not None and R is not None:
        bl["GRRI"] = G / (R + EPS)
    return bl


# =============================================================================
# Pipeline (per-task)
# =============================================================================


def run_one_task(
    csv_path: str,
    band_cols: List[str],
    label_col: str,
    target_class: str,
    block_col: str,
    C: float,
    task_index: int,
    paper_indices: bool,
    baseline_role_map: Optional[Dict[str, str]] = None,
    verbose: bool = True,
) -> Dict:
    """Run SFP on a single task/CSV and return discovered index + accuracy."""
    d = load_and_concat_csvs(
        csv_paths=[csv_path],
        band_cols=band_cols,
        label_col=label_col,
        target_class=target_class,
        block_col=block_col,
    )
    X = d[band_cols].values
    y = (d[label_col].astype(str) == target_class).astype(int).values
    block = d[block_col].values

    if verbose:
        unique_blocks, block_counts = np.unique(block, return_counts=True)
        print(f"\n--- Task {task_index + 1}: {csv_path} -----------------------")
        print(f"  Samples: {len(d):,}  Target '{target_class}' (pos): {int(y.sum()):,}  "
              f"Negative: {int((1 - y).sum()):,}")
        print(f"  Blocks: {dict(zip(unique_blocks, block_counts))}")
        print(f"  Regularization C: {C}")

    folds = two_block_folds(block)

    # Either use the paper's reference formula or run fresh SFP search
    if paper_indices:
        stage = task_index + 1
        if stage not in PAPER_WCI:
            raise ValueError(
                f"--paper-indices supports tasks 1, 2, and 3 (found task {stage})."
            )
        if len(band_cols) != 5:
            raise ValueError(
                f"--paper-indices requires exactly 5 bands in MicaSense layout "
                f"(b1=Blue, b2=Green, b3=Red, b4=RE, b5=NIR); got {len(band_cols)}."
            )
        winner_feature = PAPER_WCI[stage](X)
        winner_name = f"WCI_{stage} (paper reference formula)"
        if verbose:
            print(f"  Using paper's reference {winner_name} (no search)")
    else:
        if verbose:
            print("  Building extended basis (13 families)...")
        basis, basis_names = build_extended_basis(X, band_cols)
        if verbose:
            print(f"  {len(basis_names):,} basis features")
            print("  Running per-fold selection over full degree-2 space...")
        winner_name, winner_recipe, _, winner_feature = discover_index(
            basis, basis_names, y, folds, verbose=verbose
        )

    # Evaluate the winner with this task's C
    accs = [eval_single_feature_svm(winner_feature, y, tr, te, C=C) for tr, te in folds]
    mean_acc = float(np.mean(accs))
    gap = float(abs(accs[0] - accs[1])) if len(accs) == 2 else 0.0

    if verbose:
        print(f"  Discovered index: {winner_name}")
        if len(accs) == 2:
            print(f"    Block1->2: {accs[0] * 100:.2f}%  "
                  f"Block2->1: {accs[1] * 100:.2f}%  "
                  f"Mean: {mean_acc * 100:.2f}%  Gap: {gap * 100:.2f}")

    # Also evaluate the paper's reference (if valid) for comparison
    paper_mean = None
    if not paper_indices and len(band_cols) == 5 and (task_index + 1) in PAPER_WCI:
        pf = PAPER_WCI[task_index + 1](X)
        pa = [eval_single_feature_svm(pf, y, tr, te, C=C) for tr, te in folds]
        paper_mean = float(np.mean(pa))
        paper_gap = float(abs(pa[0] - pa[1])) if len(pa) == 2 else 0.0
        if verbose:
            print(f"  Paper reference WCI_{task_index + 1} at C={C}: "
                  f"Mean={paper_mean * 100:.2f}%  Gap={paper_gap * 100:.2f}")

    # Baselines
    baseline_rows = {}
    if baseline_role_map:
        bl = compute_baselines(d, baseline_role_map)
        if bl and verbose:
            print(f"  Baselines at C={C}:")
            print(f"    {'Index':<14s} {'Mean':>8s}  {'Gap':>6s}")
        for bname, bfeat in bl.items():
            ba = [eval_single_feature_svm(bfeat, y, tr, te, C=C) for tr, te in folds]
            baseline_rows[bname] = {
                "mean": float(np.mean(ba)),
                "gap": float(abs(ba[0] - ba[1])) if len(ba) == 2 else 0.0,
            }
            if verbose:
                print(f"    {bname:<14s} {np.mean(ba) * 100:>7.2f}% "
                      f"{abs(ba[0] - ba[1]) * 100:>5.2f}")

    return {
        "csv": csv_path,
        "winner_name": winner_name,
        "accs": accs,
        "mean_acc": mean_acc,
        "gap": gap,
        "paper_mean": paper_mean,
        "C": C,
        "baselines": baseline_rows,
    }


def run_sfp_uav(
    csv_paths: List[str],
    band_cols: List[str],
    label_col: str,
    target_class: str,
    block_col: str,
    Cs: List[float],
    paper_indices: bool = False,
    baseline_role_map: Optional[Dict[str, str]] = None,
    verbose: bool = True,
) -> Dict:
    """Run SFP on one or more UAV tasks, each with its own tuned C.

    Returns a dict of per-task results keyed by 1-based task index.
    """
    if len(Cs) == 1 and len(csv_paths) > 1:
        Cs = Cs * len(csv_paths)  # broadcast single C to all tasks
    if len(Cs) != len(csv_paths):
        raise ValueError(
            f"Number of --Cs values ({len(Cs)}) must match number of --csvs "
            f"files ({len(csv_paths)}), or be exactly 1 to apply to all."
        )

    if verbose:
        print("=" * 72)
        print("SFP UAV pipeline (Lotfi et al. 2026, MLWA)")
        print("  Basis: extended 13 families (ratios + HAbs + ...)")
        if paper_indices:
            print("  Mode: evaluating paper's reference WCI formulas")
        else:
            print("  Mode: fresh feature-space search")
        print("=" * 72)

    per_task: Dict[int, Dict] = {}
    for i, (path, C) in enumerate(zip(csv_paths, Cs)):
        per_task[i + 1] = run_one_task(
            csv_path=path,
            band_cols=band_cols,
            label_col=label_col,
            target_class=target_class,
            block_col=block_col,
            C=C,
            task_index=i,
            paper_indices=paper_indices,
            baseline_role_map=baseline_role_map,
            verbose=verbose,
        )

    if verbose:
        print("\n" + "=" * 72)
        print("Summary")
        print("=" * 72)
        print(f"  {'Task':<6s} {'Pipeline mean':>14s} {'Paper ref':>10s} "
              f"{'C':>6s}  Discovered index")
        for k, r in per_task.items():
            paper_str = f"{r['paper_mean'] * 100:>9.2f}%" if r["paper_mean"] is not None else "      n/a"
            print(f"  {k:<6d} {r['mean_acc'] * 100:>13.2f}% {paper_str} "
                  f"{r['C']:>6.3f}  {r['winner_name']}")

    return per_task


# =============================================================================
# CLI
# =============================================================================


def _parse_comma_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_role_map(s: Optional[str]) -> Optional[Dict[str, str]]:
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
            "SFP framework for calibrated UAV imagery (Lotfi et al. 2026, MLWA). "
            "Discovers one interpretable spectral index per task for a binary "
            "'target vegetation vs. everything else' classification, using the "
            "full extended basis of 13 feature families."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "\n"
            "  # Single flight, one C value\n"
            "  python sfp_uav.py --csvs myflight.csv --label-col Type --target Canopy \\\n"
            "      --bands b1,b2,b3,b4,b5 --block-col block --Cs 0.1\n"
            "\n"
            "  # Three growth stages (paper's setup), per-stage C values\n"
            "  python sfp_uav.py --csvs T1_blocked.csv T2_blocked.csv T3_blocked.csv \\\n"
            "      --label-col Type --target Canopy \\\n"
            "      --bands b1,b2,b3,b4,b5 --block-col block \\\n"
            "      --Cs 0.01,0.1,0.25\n"
            "\n"
            "  # Reproduce paper's published WCI numbers exactly\n"
            "  python sfp_uav.py --paper-indices \\\n"
            "      --csvs T1_blocked.csv T2_blocked.csv T3_blocked.csv \\\n"
            "      --label-col Type --target Canopy \\\n"
            "      --bands b1,b2,b3,b4,b5 --block-col block \\\n"
            "      --Cs 0.01,0.1,0.25\n"
        ),
    )
    ap.add_argument("--csvs", nargs="+", required=True,
                    help="One or more CSV files (one per task / flight / stage).")
    ap.add_argument("--bands", required=True,
                    help="Comma-separated band column names (e.g., b1,b2,b3,b4,b5).")
    ap.add_argument("--label-col", required=True,
                    help="Name of the categorical label column.")
    ap.add_argument("--target", required=True,
                    help="Value in --label-col that marks the target vegetation. "
                         "All other rows are 'everything else'.")
    ap.add_argument("--block-col", required=True,
                    help="Column in each CSV containing spatial-block labels "
                         "(two distinct values required, e.g., {1, 2}).")
    ap.add_argument("--Cs", required=True,
                    help="Comma-separated SVM C values, one per --csvs file "
                         "(or a single value to apply to all). Example: '0.01,0.1,0.25'.")
    ap.add_argument("--paper-indices", action="store_true",
                    help="Evaluate the paper's published WCI_1/WCI_2/WCI_3 formulas "
                         "directly instead of running a fresh feature-space search. "
                         "Only valid for MicaSense-style 5-band data "
                         "(b1=Blue, b2=Green, b3=Red, b4=RE, b5=NIR).")
    ap.add_argument("--baseline-role-map", default=None,
                    help="(Optional) comma-separated Role=Column pairs to enable "
                         "baseline comparison. Roles: Blue,Green,Red,RE,NIR. Example: "
                         "'Blue=b1,Green=b2,Red=b3,RE=b4,NIR=b5'.")
    args = ap.parse_args()

    run_sfp_uav(
        csv_paths=args.csvs,
        band_cols=_parse_comma_list(args.bands),
        label_col=args.label_col,
        target_class=args.target,
        block_col=args.block_col,
        Cs=[float(x) for x in _parse_comma_list(args.Cs)],
        paper_indices=args.paper_indices,
        baseline_role_map=_parse_role_map(args.baseline_role_map),
    )


if __name__ == "__main__":
    main()
