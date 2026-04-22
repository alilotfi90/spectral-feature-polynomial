# SFP: Spectral Feature Polynomial Framework

**Automated discovery of interpretable spectral indices for vegetation monitoring.**

> **Reference paper:** Lotfi, A., Carter, A., Ha, T., Meysami, M., Nketia, K.,
> & Shirtliffe, S. (2026). *Interpretable Machine Learning–Derived Spectral
> Indices for Vegetation Monitoring.* Machine Learning with Applications.

SFP takes a multispectral dataset in CSV form and discovers a single
closed-form algebraic index that best separates a target vegetation class
from everything else. The index is a compact algebraic function of a handful
of bands — nothing like a neural network — so it is directly interpretable,
deployable on any remote-sensing platform (including Google Earth Engine),
and requires no standardization statistics at inference time.

This repository is the reference implementation that produced the results in
the paper. It is also designed to run on your own data.

---

## Repository layout

```
sfp_core.py          Shared core (basis families, streaming ranker, CV utilities)
sfp_satellite.py     Application script for satellite imagery (invariant core basis)
sfp_uav.py           Application script for calibrated UAV imagery (extended basis)
README.md            This file
```

Keep all three `.py` files in the same directory; the two application scripts
import from `sfp_core.py`.

---

## Which script should I use?

| Use this | When you have | Because |
|---|---|---|
| `sfp_satellite.py` | Satellite imagery (Sentinel-2, Landsat, Planet, etc.) or any un-calibrated / cross-scene data | Restricts the search to illumination-invariant families (ND, ND3, NCurv). The discovered index is exactly invariant to global multiplicative scaling of reflectance — essential when scenes differ in illumination, atmosphere, or topography. |
| `sfp_uav.py` | UAV / drone imagery calibrated to surface reflectance with a pre-flight panel | Uses the full extended basis of 13 families including scale-sensitive ones (HAbs). Absolute reflectance magnitude is biologically meaningful on calibrated imagery, so the search can exploit it. |

---

## How to prepare your CSV

Both scripts share the same input convention. Your CSV should look like this:

| band_1 | band_2 | ... | band_k | label | (year) | (block) | (geometry) |
|---|---|---|---|---|---|---|---|
| 0.12 | 0.22 | ... | 0.46 | **Corn** | 2022 | 1 | {...} |
| 0.09 | 0.18 | ... | 0.41 | SoybeansRow | 2022 | 1 | {...} |
| 0.14 | 0.25 | ... | 0.48 | **Corn** | 2022 | 2 | {...} |
| 0.07 | 0.15 | ... | 0.36 | BareSoil | 2023 | 2 | {...} |
| ... | ... | ... | ... | ... | ... | ... | ... |

### What the columns mean

1. **Band columns** — one column per spectral band, holding reflectance values. The column names are arbitrary (`b1,b2,…`, `B2,B3,…`, `blue,green,red,…`, whatever you prefer); you pass the list via `--bands`. Values can be in any linear scale; normalized-difference ratios cancel the scale.

2. **Label column** — a single categorical column (any name). Pick **one value** from this column to be your **target vegetation** — the class you want an index for. Every row whose label equals that value is the positive class (`y = 1`); every row with any other value is the background / "everything else" class (`y = 0`).

   Examples:
   - To detect Kochia in a field: label values are `{Kochia, Crop}`, target is `Kochia`.
   - To detect wheat canopy versus background: label values are `{Canopy, Weed, Soil, GCP}`, target is `Canopy`.
   - To detect corn in a multi-crop field: label values are `{Corn, Soybean, Wheat, Fallow}`, target is `Corn`.
   - To detect a specific wetland type in a land-cover survey: target is that wetland label.

3. **Year column** — *optional, satellite only.* Required for leave-one-year-out cross-validation. Can hold any hashable value (integer year, date string, acquisition ID).

4. **Geometry column** — *optional, satellite only.* If you have the acquisition polygons as GeoJSON strings (one per row), provide the column name to enable spatial and spatio-temporal block cross-validation. The pipeline will derive each polygon's centroid and assign it to a 3×3 spatial grid.

5. **Block column** — *required for UAV.* Holds the spatial-block identifier (two distinct values, conventionally `1` and `2`). Draw a horizontal or vertical line across the field in your labelling tool and assign all polygons on one side to block 1 and all on the other side to block 2. This creates a physical gap between training and test partitions and prevents spatial-autocorrelation leakage, which would inflate accuracy on centimetre-scale UAV data.

---

## Installation

```bash
pip install numpy pandas scikit-learn
```

Python 3.9+ recommended. No additional packages needed.

---

## Usage

### Satellite (`sfp_satellite.py`)

**Minimal** — one CSV, random 10-fold CV, no year or geometry:

```bash
python sfp_satellite.py \
    --csvs mydata.csv \
    --label-col LandCover --target Wetland \
    --bands B2,B3,B4,B5,B6,B7,B8,B11,B12
```

**Full** — paper's Kochia reproduction (multi-year + polygons + baselines):

```bash
python sfp_satellite.py \
    --csvs year2022.csv year2023.csv year2024.csv \
    --label-col Type --target Kochia \
    --bands b1,b2,b3,b4,b5,b6,b7,b9,b10 \
    --year-col Year --geom-col .geo \
    --baseline-role-map 'Blue=b1,Green=b2,Red=b3,RE1=b4,NIR=b7,SWIR1=b9'
```

CV strategies are enabled automatically by what you provide:

| Flag present | Strategies enabled |
|---|---|
| (always) | Random 10-fold |
| `--year-col` | + Year-held-out |
| `--geom-col` | + 3×3 Spatial block |
| both | + Spatio-temporal block |

### UAV (`sfp_uav.py`)

**Single task** — one flight / one date:

```bash
python sfp_uav.py \
    --csvs myflight.csv \
    --label-col Type --target Canopy \
    --bands b1,b2,b3,b4,b5 \
    --block-col block \
    --Cs 0.1
```

**Multi-task** — paper's setup (one CSV per growth stage, per-stage `C`):

```bash
python sfp_uav.py \
    --csvs stage1.csv stage2.csv stage3.csv \
    --label-col Type --target Canopy \
    --bands b1,b2,b3,b4,b5 \
    --block-col block \
    --Cs 0.01,0.1,0.25 \
    --baseline-role-map 'Blue=b1,Green=b2,Red=b3,RE=b4,NIR=b5'
```

**Paper-reference mode** — evaluate the published WCI₁, WCI₂, WCI₃ formulas
directly (only valid for MicaSense 5-band wheat data with `b1=Blue, b2=Green,
b3=Red, b4=RE, b5=NIR`):

```bash
python sfp_uav.py --paper-indices \
    --csvs T1_blocked.csv T2_blocked.csv T3_blocked.csv \
    --label-col Type --target Canopy \
    --bands b1,b2,b3,b4,b5 \
    --block-col block \
    --Cs 0.01,0.1,0.25
```

---

## What the output looks like

### Satellite

```
========================================================================
SFP satellite pipeline (Lotfi et al. 2026, MLWA)
  Basis: illumination-invariant core families (ND, ND3, NCurv)
========================================================================

Data: 2,208 samples from 3 file(s)
  Target 'Kochia' (positive): 1,198
  Everything else (negative): 1,010
  Bands used (9): ['b1','b2','b3','b4','b5','b6','b7','b9','b10']

CV strategies enabled:
  Random 10-fold: 10 folds
  Year-held-out: 3 folds
  Spatial block: 9 folds
  Spatio-temporal: 24 folds

Building invariant-core basis...
  372 basis features (degree-2 expansion yields ~69,192 candidates)

Running per-fold selection across 46 total folds...
  Consensus: ND3[-b6+b7+b9]*NCurv[b4,b6,b7] (44/46 folds)

Accuracy of consensus index:
  Strategy              Folds     Mean   Median      Min
  Random 10-fold           10   97.46%   97.29%   95.93%
  Year-held-out             3   97.11%   98.69%   93.49%
  Spatial block             9   97.11%   99.03%   85.63%
  Spatio-temporal          24   97.36%  100.00%   70.59%
  Overall                  46   97.32%   98.89%   70.59%

Baseline comparison (mean accuracy per strategy):
  Index        Random  Year-HO  Spatial   ST
  NDVI         94.75%   92.45%   94.06%  93.51%
  NDRE         95.79%   94.10%   95.01%  94.89%
  CIre         96.06%   93.78%   95.19%  95.09%
  Discovered   97.46%   97.11%   97.11%  97.36%
```

### UAV (`--paper-indices` mode)

```
Summary
  Task   Pipeline mean  Paper ref      C  Discovered index
  1             99.45%        n/a  0.010  WCI_1 (paper reference formula)
  2             97.20%        n/a  0.100  WCI_2 (paper reference formula)
  3             93.54%        n/a  0.250  WCI_3 (paper reference formula)
```

---

## Reproducing the paper

The commands above reproduce the paper's tables to within ±0.3 percentage points:

- **Table 3** (Kochia KDI across 4 CV strategies): `sfp_satellite.py` with the paper's Kochia command.
- **Table 5** (Kochia baselines): same run, includes the `--baseline-role-map` output.
- **Section 3 wheat results** (WCI₁, WCI₂, WCI₃): `sfp_uav.py --paper-indices` with the paper's wheat command.

---

## Leakage prevention

The pipeline takes explicit precautions against data leakage:

1. **Train/test disjointness.** Every call into the SVM evaluator asserts that the training and test indices share no elements. A leak would raise an error, not fail silently.

2. **Training-only standardization.** The `StandardScaler` is fit on training rows only; test rows are transformed using the training-derived mean and standard deviation.

3. **Training-only feature ranking.** The ANOVA F-statistic used to rank candidate indices is computed on training-subset rows only. Test rows never influence which feature is selected.

4. **Spatial block integrity (UAV).** Blocks are defined at the labelling stage by a physical spatial separation in the field. Samples in different blocks are never adjacent pixels. This prevents the spatial-autocorrelation leakage that plagues random k-fold splits on high-resolution drone imagery.

5. **Per-fold consensus.** The final "winning" index is chosen as the feature most frequently selected across folds, where each fold's selection depends only on its own training data. This is the inner-loop analogue of nested cross-validation.

---

## Methodology summary

For every fold:

1. Build the basis features on the training rows (the basis is deterministic per-row, so no leak).
2. For every degree-2 candidate (basis term, squared basis term, pairwise product of basis terms), compute the ANOVA F-statistic on training rows.
3. Pick the candidate with the highest F-statistic as that fold's selected feature.
4. Train a linear SVM on that single feature using standardized training rows; evaluate on the held-out test rows.

The *consensus winner* across folds is reported as the discovered index. Its accuracy is reported per CV strategy.

### Memory efficiency

The full degree-2 space of a 372-feature basis has ~70,000 candidates. Rather than materializing this as one dense matrix (which would require several gigabytes for thousands of samples), the ranker streams columns in batches: for each basis column `i`, it computes `basis[:, i+1:] * basis[:, i:i+1]` on the fly, scores each column, and frees the batch. Peak memory stays at O(n_samples × n_basis).

---

## Citation

```bibtex
@article{lotfi2026sfp,
  title   = {Interpretable Machine Learning-Derived Spectral Indices for Vegetation Monitoring},
  author  = {Lotfi, Ali and Carter, Adam and Ha, Thuan and
             Meysami, Mohammad and Nketia, Kwabena and Shirtliffe, Steve},
  journal = {Machine Learning with Applications},
  year    = {2026}
}
```

## License

MIT.

## Acknowledgments

Supported by the Saskatchewan Ministry of Agriculture through the Agriculture
Development Fund (grant number 20230164).

## Contact

Ali Lotfi, Ph.D. — Postdoctoral Fellow, Nutrien Centre for Sustainable and
Digital Agriculture, University of Saskatchewan, Saskatoon, SK, Canada.
`all054@usask.ca`
