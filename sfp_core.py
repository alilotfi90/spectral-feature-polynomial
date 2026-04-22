#!/usr/bin/env python3
"""
sfp_core.py
===========

Shared core utilities for the Spectral Feature Polynomial (SFP) framework.

This module implements the reusable components of the SFP pipeline — the
basis families, the memory-efficient streaming feature ranker, the
cross-validation utilities, and the SVM evaluator — that are invoked by the
application-specific orchestration scripts (`sfp_satellite.py` for Sentinel-2
satellite data and `sfp_uav.py` for calibrated UAV imagery).

Both scripts import from this module rather than duplicating the logic.

Reference
---------
Lotfi, A., Carter, A., Ha, T., Meysami, M., Nketia, K., & Shirtliffe, S. (2026).
Interpretable Machine Learning-Derived Spectral Indices for Vegetation
Monitoring. Machine Learning with Applications.

License: MIT
"""

from __future__ import annotations

import json
import warnings
from collections import Counter
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

EPS = 1e-10

# =============================================================================
# Core invariant basis families (ND, ND3, NCurv)
#
# These three families are exactly invariant to global multiplicative scaling
# of reflectance: for any c > 0, f(c*b) = f(b). That property is why the
# satellite pipeline restricts itself to this subset — satellite scenes differ
# in illumination, atmospheric attenuation, and view geometry, and exact
# scale invariance eliminates those effects from the index output.
# =============================================================================


def build_nd(X: np.ndarray, bands: List[str]) -> Tuple[List[np.ndarray], List[str]]:
    """Two-band normalized difference: (b_i - b_j) / (b_i + b_j + eps)."""
    n = len(bands)
    feats, names = [], []
    for i, j in combinations(range(n), 2):
        feats.append((X[:, i] - X[:, j]) / (X[:, i] + X[:, j] + EPS))
        names.append(f"ND[{bands[i]},{bands[j]}]")
    return feats, names


def build_nd3(X: np.ndarray, bands: List[str]) -> Tuple[List[np.ndarray], List[str]]:
    """Three-band normalized difference with one negative sign."""
    n = len(bands)
    feats, names = [], []
    for i, j, k in combinations(range(n), 3):
        den = X[:, i] + X[:, j] + X[:, k] + EPS
        for neg_idx in range(3):
            signs = [1, 1, 1]
            signs[neg_idx] = -1
            num = signs[0] * X[:, i] + signs[1] * X[:, j] + signs[2] * X[:, k]
            feats.append(num / den)
            s = ["+" if x > 0 else "-" for x in signs]
            names.append(
                f"ND3[{s[0]}{bands[i]}{s[1]}{bands[j]}{s[2]}{bands[k]}]"
            )
    return feats, names


def build_ncurv(X: np.ndarray, bands: List[str]) -> Tuple[List[np.ndarray], List[str]]:
    """Normalized curvature: (b_i - 2*b_j + b_k) / (b_i + 2*b_j + b_k + eps)."""
    n = len(bands)
    feats, names = [], []
    for i, j, k in combinations(range(n), 3):
        feats.append(
            (X[:, i] - 2 * X[:, j] + X[:, k])
            / (X[:, i] + 2 * X[:, j] + X[:, k] + EPS)
        )
        names.append(f"NCurv[{bands[i]},{bands[j]},{bands[k]}]")
    return feats, names


def build_core_basis(
    X: np.ndarray, bands: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """Concatenate the three illumination-invariant core families.

    Use this for satellite imagery where illumination invariance is required.
    """
    feats, names = [], []
    for fn in (build_nd, build_nd3, build_ncurv):
        f, n = fn(X, bands)
        feats.extend(f)
        names.extend(n)
    return np.column_stack(feats), names


# =============================================================================
# Extended basis families (10 additional, 13 total)
#
# These include scale-sensitive families (particularly HAbs) that require
# calibrated reflectance inputs. Use the extended basis for UAV / drone data
# that has been radiometrically calibrated with a pre-flight reference panel.
# =============================================================================


def build_sr(X, bands):
    """Bounded simple ratio: tanh(b_i / b_j - 1), ordered pairs."""
    n = len(bands)
    feats, names = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            feats.append(np.tanh(X[:, i] / (X[:, j] + EPS) - 1))
            names.append(f"SR[{bands[i]}/{bands[j]}]")
    return feats, names


def build_bandratio(X, bands):
    """(b_i - max_{j!=i} b_j) / (b_i + max_{j!=i} b_j + eps)."""
    n = len(bands)
    feats, names = [], []
    for i in range(n):
        others = [k for k in range(n) if k != i]
        m = X[:, others].max(axis=1)
        feats.append((X[:, i] - m) / (X[:, i] + m + EPS))
        names.append(f"BandRatio[{bands[i]}]")
    return feats, names


def build_bop(X, bands):
    """BOP: b_i / (b_j + b_k + eps) with i not in {j,k}."""
    n = len(bands)
    feats, names = [], []
    for i in range(n):
        for j, k in combinations([x for x in range(n) if x != i], 2):
            feats.append(X[:, i] / (X[:, j] + X[:, k] + EPS))
            names.append(f"BOP[{bands[i]}/({bands[j]}+{bands[k]})]")
    return feats, names


def build_bot(X, bands):
    """BOT: b_i / (b_j + b_k + b_l + eps) with i not in {j,k,l}."""
    n = len(bands)
    feats, names = [], []
    for i in range(n):
        for j, k, l in combinations([x for x in range(n) if x != i], 3):
            feats.append(X[:, i] / (X[:, j] + X[:, k] + X[:, l] + EPS))
            names.append(f"BOT[{bands[i]}/({bands[j]}+{bands[k]}+{bands[l]})]")
    return feats, names


def build_boa(X, bands):
    """BOA: b_i / (sum_{j!=i} b_j + eps)."""
    n = len(bands)
    feats, names = [], []
    total = X.sum(axis=1)
    for i in range(n):
        feats.append(X[:, i] / (total - X[:, i] + EPS))
        names.append(f"BOA[{bands[i]}]")
    return feats, names


def build_bfrac(X, bands):
    """BFrac: b_i / sum_j b_j."""
    n = len(bands)
    feats, names = [], []
    total = X.sum(axis=1) + EPS
    for i in range(n):
        feats.append(X[:, i] / total)
        names.append(f"BFrac[{bands[i]}]")
    return feats, names


def build_sr2(X, bands):
    """SR2: (b_i + b_j) / (b_k + b_l + eps) with {i,j} and {k,l} disjoint."""
    n = len(bands)
    feats, names = [], []
    all_pairs = list(combinations(range(n), 2))
    for (i, j) in all_pairs:
        for (k, l) in all_pairs:
            if (i, j) == (k, l):
                continue
            if len({i, j, k, l}) < 4:
                continue
            feats.append((X[:, i] + X[:, j]) / (X[:, k] + X[:, l] + EPS))
            names.append(f"SR2[({bands[i]}+{bands[j]})/({bands[k]}+{bands[l]})]")
    return feats, names


def build_sr3(X, bands):
    """SR3: (b_i + b_j + b_k) / b_l with l not in {i,j,k}."""
    n = len(bands)
    feats, names = [], []
    for i, j, k in combinations(range(n), 3):
        for l in range(n):
            if l in (i, j, k):
                continue
            feats.append((X[:, i] + X[:, j] + X[:, k]) / (X[:, l] + EPS))
            names.append(f"SR3[({bands[i]}+{bands[j]}+{bands[k]})/{bands[l]}]")
    return feats, names


def build_hsurr(X, bands):
    """HSurr: tanh((b_i - max_{j!=i} b_j) / mean_{j!=i} b_j). Scale-invariant."""
    n = len(bands)
    feats, names = [], []
    for i in range(n):
        others_idx = [k for k in range(n) if k != i]
        m = X[:, others_idx].max(axis=1)
        mu = X[:, others_idx].mean(axis=1)
        feats.append(np.tanh((X[:, i] - m) / (mu + EPS)))
        names.append(f"HSurr[{bands[i]}]")
    return feats, names


def build_habs(X, bands):
    """HAbs: tanh(b_i - max_{j!=i} b_j). Scale-SENSITIVE (requires calibrated reflectance)."""
    n = len(bands)
    feats, names = [], []
    for i in range(n):
        others_idx = [k for k in range(n) if k != i]
        m = X[:, others_idx].max(axis=1)
        feats.append(np.tanh(X[:, i] - m))
        names.append(f"HAbs[{bands[i]}]")
    return feats, names


def build_speccv(X, bands):
    """SpecCV: coefficient of variation across all bands."""
    return [X.std(axis=1) / (X.mean(axis=1) + EPS)], ["SpecCV"]


def build_extended_basis(
    X: np.ndarray, bands: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """All 13 basis families, including scale-sensitive ones.

    Use this for calibrated UAV imagery where absolute reflectance magnitudes
    are biologically meaningful and strict illumination invariance is not
    required.
    """
    feats, names = [], []
    for fn in (
        build_nd, build_nd3, build_ncurv,
        build_sr, build_bandratio,
        build_bop, build_bot, build_boa, build_bfrac,
        build_sr2, build_sr3,
        build_hsurr, build_habs, build_speccv,
    ):
        f, n = fn(X, bands)
        feats.extend(f)
        names.extend(n)
    return np.column_stack(feats), names


# =============================================================================
# Memory-efficient degree-2 polynomial ranking
# =============================================================================


def rank_degree2_features_streaming(
    basis: np.ndarray, basis_names: List[str], y: np.ndarray
) -> List[Tuple[float, str, Tuple]]:
    """Rank every degree-2 candidate by ANOVA F-statistic, streaming per batch.

    Avoids materializing the full (n_samples, n_basis^2) product matrix by
    scoring feature columns in batches. Returns a list of records
        (F_stat, feature_name, feature_recipe)
    sorted descending, where `feature_recipe` is a tuple that `materialize_feature`
    can use to rebuild the exact feature column.

    Leakage note: this function must be called with TRAINING rows only
    (basis[train_idx], y[train_idx]). Test rows must not influence
    feature ranking. The caller is responsible for passing the correct subset.
    """
    n_basis = basis.shape[1]
    records: List[Tuple[float, str, Tuple]] = []

    # Basis terms themselves
    F_basis, _ = f_classif(basis, y)
    F_basis = np.nan_to_num(F_basis, nan=0.0, posinf=0.0, neginf=0.0)
    for i in range(n_basis):
        records.append((float(F_basis[i]), basis_names[i], ("basis", i)))

    # Squares
    sq = basis ** 2
    F_sq, _ = f_classif(sq, y)
    F_sq = np.nan_to_num(F_sq, nan=0.0, posinf=0.0, neginf=0.0)
    for i in range(n_basis):
        records.append((float(F_sq[i]), f"({basis_names[i]})^2", ("square", i)))

    # Pairwise products, batched by anchor column
    for i in range(n_basis):
        if i >= n_basis - 1:
            break
        cols = basis[:, i : i + 1] * basis[:, i + 1 :]
        F_cols, _ = f_classif(cols, y)
        F_cols = np.nan_to_num(F_cols, nan=0.0, posinf=0.0, neginf=0.0)
        for k, j in enumerate(range(i + 1, n_basis)):
            records.append(
                (float(F_cols[k]), f"{basis_names[i]}*{basis_names[j]}",
                 ("product", i, j))
            )

    records.sort(key=lambda r: r[0], reverse=True)
    return records


def materialize_feature(basis: np.ndarray, recipe: Tuple) -> np.ndarray:
    """Rebuild a feature column from its recipe tuple."""
    if recipe[0] == "basis":
        return basis[:, recipe[1]]
    if recipe[0] == "square":
        return basis[:, recipe[1]] ** 2
    if recipe[0] == "product":
        return basis[:, recipe[1]] * basis[:, recipe[2]]
    raise ValueError(f"Unknown recipe: {recipe}")


# =============================================================================
# SVM evaluator (with explicit leak-prevention)
# =============================================================================


def eval_single_feature_svm(
    feature: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    C: float = 0.5,
) -> float:
    """Train a linear SVM on a single feature and return held-out test accuracy.

    Leakage prevention:
      * train_idx and test_idx are asserted disjoint
      * StandardScaler is fit on training rows only; test rows are transformed
        using the training-derived mean and standard deviation
    """
    assert np.intersect1d(train_idx, test_idx).size == 0, \
        "train_idx and test_idx overlap — data leak!"
    Xtr = feature[train_idx].reshape(-1, 1)
    Xte = feature[test_idx].reshape(-1, 1)
    scaler = StandardScaler().fit(Xtr)
    clf = SVC(kernel="linear", C=C)
    clf.fit(scaler.transform(Xtr), y[train_idx])
    return clf.score(scaler.transform(Xte), y[test_idx])


# =============================================================================
# Cross-validation strategies
# =============================================================================


def polygon_centroid(geo_str: str) -> Tuple[float, float]:
    """Centroid of a GeoJSON Polygon string."""
    try:
        g = json.loads(geo_str)
        ring = g["coordinates"][0]
        return (
            float(np.mean([p[0] for p in ring])),
            float(np.mean([p[1] for p in ring])),
        )
    except Exception:
        return np.nan, np.nan


def assign_spatial_blocks(
    lons: np.ndarray, lats: np.ndarray, grid: int = 3
) -> np.ndarray:
    """Assign each sample to a grid x grid spatial block using lon/lat tertiles."""
    q = np.linspace(0, 1, grid + 1)[1:-1]
    lon_q = np.quantile(lons[~np.isnan(lons)], q)
    lat_q = np.quantile(lats[~np.isnan(lats)], q)
    col = np.digitize(lons, lon_q)
    row = np.digitize(lats, lat_q)
    return row * grid + col


def random_kfold(y: np.ndarray, n_splits: int = 10, seed: int = 42):
    """Stratified k-fold folds. Stratifies to preserve class balance per fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros_like(y), y))


def year_held_out_folds(years: np.ndarray, y: Optional[np.ndarray] = None):
    """Leave-one-year-out CV using distinct values in `years`.

    Skips folds with empty train sets (happens with single-year data) or
    single-class test sets. Returns an empty list if no valid fold exists
    (e.g., only one year in the data).
    """
    folds = []
    for yr in sorted(np.unique(years)):
        train = np.where(years != yr)[0]
        test = np.where(years == yr)[0]
        if len(train) == 0 or len(test) == 0:
            continue
        if y is not None and len(np.unique(y[test])) < 2:
            continue
        folds.append((train, test))
    return folds


def block_held_out_folds(block: np.ndarray, y: np.ndarray, min_test: int = 5):
    """Leave-one-block-out CV; skip folds with too few or single-class test sets."""
    folds = []
    for b in np.unique(block):
        test = np.where(block == b)[0]
        train = np.where(block != b)[0]
        if len(test) < min_test or len(np.unique(y[test])) < 2:
            continue
        folds.append((train, test))
    return folds


def spatiotemporal_folds(
    years: np.ndarray, block: np.ndarray, y: np.ndarray, min_test: int = 5
):
    """Leave-one-{year, block}-out CV, skipping thin or single-class test sets."""
    folds = []
    for yr in np.unique(years):
        for b in np.unique(block):
            mask = (years == yr) & (block == b)
            test = np.where(mask)[0]
            train = np.where(~mask)[0]
            if len(test) < min_test or len(np.unique(y[test])) < 2:
                continue
            folds.append((train, test))
    return folds


def two_block_folds(block: np.ndarray):
    """Two-fold CV in Block1->2 then Block2->1 order (for UAV-style designs).

    Fold 0: train on first-labeled block, test on second  (e.g., 1 -> 2)
    Fold 1: train on second-labeled block, test on first  (e.g., 2 -> 1)

    Raises a ValueError with guidance if the block column does not have
    exactly two distinct values.
    """
    unique_blocks = sorted(np.unique(block))
    if len(unique_blocks) != 2:
        raise ValueError(
            f"Two-fold spatial CV requires exactly 2 distinct values in the "
            f"block column, but found {len(unique_blocks)}: {unique_blocks}. "
            f"Split your samples into two spatially separated blocks before "
            f"running the pipeline (see README: 'How to prepare your CSV')."
        )
    folds = []
    # Fold 0 tests the later block, fold 1 tests the earlier: this matches the
    # paper's "train on Block 1, test on Block 2 then reversed" ordering.
    for test_b in reversed(unique_blocks):
        train = np.where(block != test_b)[0]
        test = np.where(block == test_b)[0]
        assert np.intersect1d(train, test).size == 0
        folds.append((train, test))
    return folds


# =============================================================================
# SFP pipeline: per-fold selection + consensus winner
# =============================================================================


def discover_index(
    basis: np.ndarray,
    basis_names: List[str],
    y: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    verbose: bool = True,
) -> Tuple[str, Tuple, Counter, np.ndarray]:
    """Run per-fold SFP selection and return the consensus winner.

    For each fold, the highest-F-statistic degree-2 feature is selected using
    training rows only. The consensus winner is the feature selected most
    often across folds.

    Returns
    -------
    winner_name : str
    winner_recipe : tuple
    selection_counter : Counter of feature_name -> fold_count
    winner_feature : (n_samples,) array, the full-data realization of the winner
    """
    selection_counter: Counter = Counter()
    recipe_by_name: Dict[str, Tuple] = {}

    if len(folds) == 0:
        raise ValueError(
            "discover_index was called with zero folds. This usually means "
            "every cross-validation strategy was skipped (e.g., single year "
            "with no spatial structure). Provide a --year-col or --geom-col "
            "argument, or ensure your data has enough structure for at least "
            "Random 10-fold to run."
        )

    for tr, te in folds:
        ranked = rank_degree2_features_streaming(basis[tr], basis_names, y[tr])
        _, top_name, top_recipe = ranked[0]
        selection_counter[top_name] += 1
        recipe_by_name.setdefault(top_name, top_recipe)

    winner_name, winner_count = selection_counter.most_common(1)[0]
    winner_recipe = recipe_by_name[winner_name]
    winner_feature = materialize_feature(basis, winner_recipe)

    if verbose:
        total = len(folds)
        print(f"  Consensus: {winner_name} ({winner_count}/{total} folds)")
        if len(selection_counter) > 1:
            for nm, c in list(selection_counter.most_common(5))[1:]:
                print(f"    also: {nm} ({c}/{total} folds)")

    return winner_name, winner_recipe, selection_counter, winner_feature


# =============================================================================
# Data loading with column validation
# =============================================================================


def load_and_concat_csvs(
    csv_paths: List[str],
    band_cols: List[str],
    label_col: str,
    target_class: str,
    year_col: Optional[str] = None,
    block_col: Optional[str] = None,
    geom_col: Optional[str] = None,
) -> pd.DataFrame:
    """Load one or more CSV files, validate required columns, and concatenate.

    A binary target column `y` is derived by comparing `label_col` against
    `target_class`: rows where the label equals the target are positive (1),
    all others are negative (0). This is the "target vegetation vs. everything
    else" formulation described in the README.

    Parameters
    ----------
    csv_paths : list of paths
    band_cols : names of the band (reflectance) columns in the CSV
    label_col : name of the categorical label column
    target_class : the value in `label_col` that defines the positive class
    year_col, block_col, geom_col : optional auxiliary columns
    """
    dfs = []
    for i, path in enumerate(csv_paths):
        df = pd.read_csv(path)
        needed = list(band_cols) + [label_col]
        for c in (year_col, block_col, geom_col):
            if c is not None:
                needed.append(c)
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(
                f"File {path!r} is missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )
        # If user gave multiple files but no year column, synthesize one
        if year_col is None and len(csv_paths) > 1:
            df["__source__"] = i
        dfs.append(df)
    d = pd.concat(dfs, ignore_index=True)

    # Check the target class exists
    if target_class not in set(d[label_col].astype(str)):
        raise ValueError(
            f"target_class={target_class!r} not found in column {label_col!r}. "
            f"Unique values: {sorted(set(d[label_col].astype(str)))}"
        )
    return d
