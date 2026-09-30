#!/usr/bin/env python3
"""
cross_validate_ocr.py
======================
Stratified k-fold cross-validation for the HOG + RBF-SVM OCR model used in the
MSA36HN Group 10 Vietnamese License Plate Recognition project.

WHY THIS SCRIPT EXISTS
----------------------
The report's controlled OCR accuracy (e.g. 95.64%) comes from a *single*
stratified 80/20 hold-out. A single split can be optimistic or pessimistic by
chance. This script runs stratified k-fold CV and reports `mean +/- std` for
accuracy and macro-F1, plus a 95% confidence interval of the mean. Paste the
output straight into the report's cross-validation table (Table "tab:cv").

IT MIRRORS TRAINING EXACTLY
---------------------------
- It recovers the HOG configuration and SVM hyper-parameters from the trained
  model directory (metadata.json / the saved estimator) so CV uses the same
  settings as training. Falls back to documented defaults if metadata is absent
  (HOG324: 32x32, 8x8 cells, 2x2 blocks, 9 orientations, L2-Hys; SVC C=10,
  gamma="scale", RBF, one-vs-one).
- StandardScaler is placed INSIDE the CV pipeline, so it is re-fit on each
  training fold only. This prevents scaling leakage, matching the report.

TWO WAYS TO PROVIDE DATA (pick whichever you have)
--------------------------------------------------
Mode A  --features PATH        Cached feature matrix (fastest, exact).
        Accepts: an .npz with arrays {X, y}, or a directory/.npy pair.
        This is the recommended mode: add the snippet in `--help-cache`
        to train_ocr_model.py once, re-run training, then point here.

Mode B  --data-dir PATH        Class-folder image dataset. Layout:
            data-dir/<class_label>/<anything>.png|jpg|...
        The script loads each crop as grayscale, resizes to the HOG input size,
        extracts HOG (matching the recovered config) and runs CV.
        IMPORTANT: to reproduce the reported number you must point this at the
        *same assembled training set* used by train_ocr_model.py (generic
        synthetic + plate-style synthetic + Auto20 real crops), not only the
        Auto20 real subset. The Auto20-only folder will give a different,
        smaller-sample number.

EXAMPLES
--------
    # Mode A: cached features saved by training
    python scripts/cross_validate_ocr.py \
        --model-dir models/ocr_eval_hog324_auto20_no_emnist \
        --features  models/ocr_eval_hog324_auto20_no_emnist/features.npz \
        --folds 5 --seed 42 \
        --output results/cv_hog324_auto20.csv

    # Mode B: rebuild HOG from a class-folder image set
    python scripts/cross_validate_ocr.py \
        --model-dir models/ocr_eval_hog324_auto20_no_emnist \
        --data-dir  data/characters/ocr_training_assembled \
        --folds 5 --seed 42 \
        --output results/cv_hog324_auto20.csv

    # Compare several trained models in one run (repeat --model-dir/--features)
    python scripts/cross_validate_ocr.py \
        --model-dir models/ocr_eval_hog324_auto20_no_emnist --features .../features.npz \
        --model-dir models/ocr_eval_hog_legacy_no_emnist    --features .../features.npz \
        --folds 5 --seed 42 --output results/cv_all.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# --- sklearn / skimage are required ---------------------------------------
try:
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
except Exception as exc:  # pragma: no cover
    sys.exit(f"[fatal] scikit-learn is required: {exc}\n  pip install scikit-learn")

try:
    from scipy import stats as _scipy_stats

    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ===========================================================================
# Configuration recovered from a trained model directory
# ===========================================================================
@dataclass
class HogConfig:
    """HOG settings. Defaults reproduce HOG324 on a 32x32 crop."""

    image_size: tuple[int, int] = (32, 32)
    pixels_per_cell: tuple[int, int] = (8, 8)
    cells_per_block: tuple[int, int] = (2, 2)
    orientations: int = 9
    block_norm: str = "L2-Hys"
    transform_sqrt: bool = False
    binarize: bool = False  # set True only if training binarized before HOG

    @property
    def expected_dim(self) -> int:
        h, w = self.image_size
        cph = h // self.pixels_per_cell[0]
        cpw = w // self.pixels_per_cell[1]
        bph = cph - self.cells_per_block[0] + 1
        bpw = cpw - self.cells_per_block[1] + 1
        return (
            bph
            * bpw
            * self.cells_per_block[0]
            * self.cells_per_block[1]
            * self.orientations
        )


@dataclass
class SvmConfig:
    """SVM hyper-parameters. Defaults match the report."""

    C: float = 10.0
    kernel: str = "rbf"
    gamma: str | float = "scale"
    decision_function_shape: str = "ovo"
    class_weight: Optional[str] = None


@dataclass
class ModelSpec:
    name: str
    model_dir: Optional[Path]
    hog: HogConfig = field(default_factory=HogConfig)
    svm: SvmConfig = field(default_factory=SvmConfig)


# ===========================================================================
# Recover config from model dir (metadata.json and/or the saved estimator)
# ===========================================================================
_METADATA_NAMES = ("metadata.json", "meta.json", "model_metadata.json", "config.json")
_MODEL_NAMES = (
    "model.joblib",
    "classifier.joblib",
    "svm.joblib",
    "ocr_model.joblib",
    "model.pkl",
    "classifier.pkl",
    "svm.pkl",
)


def _load_json(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _maybe(d: dict, *keys, default=None):
    """Return first present key (case-insensitive, nested-friendly)."""
    low = {k.lower(): v for k, v in d.items()} if isinstance(d, dict) else {}
    for k in keys:
        if k.lower() in low:
            return low[k.lower()]
    return default


def recover_config(
    model_dir: Optional[Path],
    cli_pixels_per_cell: Optional[int],
    cli_binarize: Optional[bool],
) -> tuple[HogConfig, SvmConfig]:
    hog = HogConfig()
    svm = SvmConfig()

    meta = {}
    if model_dir is not None and model_dir.exists():
        for nm in _METADATA_NAMES:
            p = model_dir / nm
            if p.exists():
                meta = _load_json(p)
                if meta:
                    print(f"[info] loaded metadata: {p}")
                    break

    if meta:
        # HOG block (accept a few key spellings)
        hb = _maybe(meta, "hog", "feature", "features", default={}) or {}
        if isinstance(hb, dict):
            ppc = _maybe(hb, "pixels_per_cell", "cell_size", "cell")
            if isinstance(ppc, (list, tuple)) and len(ppc) == 2:
                hog.pixels_per_cell = (int(ppc[0]), int(ppc[1]))
            elif isinstance(ppc, int):
                hog.pixels_per_cell = (ppc, ppc)
            cpb = _maybe(hb, "cells_per_block", "block_size", "block")
            if isinstance(cpb, (list, tuple)) and len(cpb) == 2:
                hog.cells_per_block = (int(cpb[0]), int(cpb[1]))
            ori = _maybe(hb, "orientations", "bins", "nbins")
            if isinstance(ori, int):
                hog.orientations = ori
            isz = _maybe(hb, "image_size", "input_size", "size")
            if isinstance(isz, (list, tuple)) and len(isz) == 2:
                hog.image_size = (int(isz[0]), int(isz[1]))
            elif isinstance(isz, int):
                hog.image_size = (isz, isz)
            bn = _maybe(hb, "block_norm")
            if isinstance(bn, str):
                hog.block_norm = bn
            bz = _maybe(hb, "binarize", "binary")
            if isinstance(bz, bool):
                hog.binarize = bz

        # A "feature method" string hint: hog_legacy == HOG324 (8x8)
        fmethod = str(_maybe(meta, "feature_method", "feature", default="")).lower()
        if "1764" in fmethod:
            hog.pixels_per_cell = (4, 4)
        elif "legacy" in fmethod or "324" in fmethod:
            hog.pixels_per_cell = (8, 8)

        # SVM block
        sb = _maybe(meta, "svm", "classifier", "clf", default={}) or {}
        if isinstance(sb, dict):
            if _maybe(sb, "C") is not None:
                svm.C = float(_maybe(sb, "C"))
            if _maybe(sb, "kernel") is not None:
                svm.kernel = str(_maybe(sb, "kernel"))
            if _maybe(sb, "gamma") is not None:
                g = _maybe(sb, "gamma")
                svm.gamma = g if isinstance(g, str) else float(g)

    # Try to read params straight off the saved estimator (most authoritative)
    if model_dir is not None and model_dir.exists():
        est = _try_load_estimator(model_dir)
        if est is not None:
            params = _extract_svc_params(est)
            if params:
                svm.C = params.get("C", svm.C)
                svm.kernel = params.get("kernel", svm.kernel)
                svm.gamma = params.get("gamma", svm.gamma)
                print(
                    f"[info] recovered SVM params from estimator: "
                    f"C={svm.C}, kernel={svm.kernel}, gamma={svm.gamma}"
                )

    # CLI overrides win
    if cli_pixels_per_cell is not None:
        hog.pixels_per_cell = (cli_pixels_per_cell, cli_pixels_per_cell)
    if cli_binarize is not None:
        hog.binarize = cli_binarize

    return hog, svm


def _try_load_estimator(model_dir: Path):
    try:
        import joblib
    except Exception:
        return None
    for nm in _MODEL_NAMES:
        p = model_dir / nm
        if p.exists():
            try:
                obj = joblib.load(p)
                print(f"[info] loaded estimator: {p}")
                return obj
            except Exception as exc:
                print(f"[warn] could not load {p}: {exc}")
    return None


def _extract_svc_params(obj) -> dict:
    """Find an SVC inside a saved object (estimator, pipeline, or dict) and
    return its key params."""
    cand = obj
    if isinstance(obj, dict):
        for k in ("model", "classifier", "svm", "clf", "estimator"):
            if k in obj:
                cand = obj[k]
                break
    # Pipeline?
    steps = getattr(cand, "named_steps", None)
    if steps:
        for v in steps.values():
            if isinstance(v, SVC):
                cand = v
                break
    if isinstance(cand, SVC):
        return {"C": cand.C, "kernel": cand.kernel, "gamma": cand.gamma}
    return {}


# ===========================================================================
# Data loading
# ===========================================================================
def load_cached_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load X, y from .npz, or from X.npy + y.npy in a directory."""
    if path.is_dir():
        xp = next(
            (
                path / n
                for n in ("X.npy", "features.npy", "hog.npy")
                if (path / n).exists()
            ),
            None,
        )
        yp = next(
            (
                path / n
                for n in ("y.npy", "labels.npy", "targets.npy")
                if (path / n).exists()
            ),
            None,
        )
        if xp is None or yp is None:
            raise FileNotFoundError(f"Could not find X/y .npy files in {path}")
        return np.load(xp), np.load(yp, allow_pickle=True)
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        xkey = next((k for k in ("X", "features", "hog", "x") if k in data), None)
        ykey = next((k for k in ("y", "labels", "targets", "label") if k in data), None)
        if xkey is None or ykey is None:
            raise KeyError(
                f"{path} must contain arrays named X and y "
                f"(found keys: {list(data.keys())})"
            )
        return data[xkey], data[ykey]
    if path.suffix == ".npy":
        # assume sibling y.npy
        yp = path.with_name("y.npy")
        if not yp.exists():
            raise FileNotFoundError(f"Expected labels at {yp}")
        return np.load(path), np.load(yp, allow_pickle=True)
    raise ValueError(f"Unsupported features path: {path}")


_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm"}


def build_features_from_images(
    data_dir: Path, hog: HogConfig, verbose: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Walk a class-folder dataset and extract HOG features."""
    try:
        from skimage.feature import hog as sk_hog
        from skimage.io import imread
        from skimage.transform import resize
        from skimage.color import rgb2gray
        from skimage.filters import threshold_otsu
    except Exception as exc:  # pragma: no cover
        sys.exit(
            f"[fatal] scikit-image is required for --data-dir mode: {exc}\n"
            f"  pip install scikit-image"
        )

    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(f"No class subfolders found under {data_dir}")

    X, y = [], []
    t0 = time.time()
    total = 0
    for cdir in class_dirs:
        label = cdir.name
        files = [p for p in cdir.rglob("*") if p.suffix.lower() in _IMG_EXTS]
        for fp in files:
            try:
                img = imread(fp)
            except Exception:
                continue
            if img.ndim == 3:
                img = rgb2gray(img)
            img = img.astype(np.float64)
            mx = img.max()
            if mx > 1.0:
                img = img / 255.0
            img = resize(img, hog.image_size, anti_aliasing=True)
            if hog.binarize:
                try:
                    thr = threshold_otsu(img)
                    img = (img > thr).astype(np.float64)
                except Exception:
                    pass
            feat = sk_hog(
                img,
                orientations=hog.orientations,
                pixels_per_cell=hog.pixels_per_cell,
                cells_per_block=hog.cells_per_block,
                block_norm=hog.block_norm,
                transform_sqrt=hog.transform_sqrt,
                feature_vector=True,
            )
            X.append(feat)
            y.append(label)
            total += 1
        if verbose:
            print(f"[load] {label:>3}: {len(files):5d} files  (cumulative {total})")
    if not X:
        raise RuntimeError(f"No images loaded from {data_dir}")
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if verbose:
        print(
            f"[load] built {X.shape[0]} samples x {X.shape[1]} dims "
            f"in {time.time()-t0:.1f}s"
        )
    return X, y


# ===========================================================================
# Cross-validation
# ===========================================================================
def mean_ci95(values: np.ndarray) -> tuple[float, float, float, float]:
    """Return (mean, std, ci_low, ci_high) for the mean across folds.
    Uses Student-t for the small number of folds; falls back to normal."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    m = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        sem = sd / math.sqrt(n)
        if _HAVE_SCIPY:
            tcrit = float(_scipy_stats.t.ppf(0.975, df=n - 1))
        else:
            tcrit = 1.96
        half = tcrit * sem
    else:
        half = 0.0
    return m, sd, m - half, m + half


def run_cv(X: np.ndarray, y: np.ndarray, svm: SvmConfig, folds: int, seed: int) -> dict:
    pipe = make_pipeline(
        StandardScaler(),
        SVC(
            C=svm.C,
            kernel=svm.kernel,
            gamma=svm.gamma,
            decision_function_shape=svm.decision_function_shape,
        ),
    )
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scoring = {"accuracy": "accuracy", "macro_f1": "f1_macro"}
    t0 = time.time()
    res = cross_validate(
        pipe, X, y, cv=skf, scoring=scoring, n_jobs=-1, return_train_score=False
    )
    elapsed = time.time() - t0

    acc = res["test_accuracy"]
    f1 = res["test_macro_f1"]
    a_m, a_sd, a_lo, a_hi = mean_ci95(acc)
    f_m, f_sd, f_lo, f_hi = mean_ci95(f1)
    return {
        "folds": folds,
        "seed": seed,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(np.unique(y))),
        "acc_per_fold": acc.tolist(),
        "f1_per_fold": f1.tolist(),
        "acc_mean": a_m,
        "acc_std": a_sd,
        "acc_ci": (a_lo, a_hi),
        "f1_mean": f_m,
        "f1_std": f_sd,
        "f1_ci": (f_lo, f_hi),
        "elapsed_s": elapsed,
    }


# ===========================================================================
# Reporting
# ===========================================================================
def print_result(name: str, r: dict) -> None:
    print("\n" + "=" * 68)
    print(f"  {name}")
    print("=" * 68)
    print(
        f"  samples={r['n_samples']}  features={r['n_features']}  "
        f"classes={r['n_classes']}  folds={r['folds']}  seed={r['seed']}"
    )
    print(
        f"  per-fold accuracy : "
        f"{', '.join(f'{x*100:.2f}' for x in r['acc_per_fold'])}"
    )
    print(
        f"  per-fold macro-F1 : "
        f"{', '.join(f'{x*100:.2f}' for x in r['f1_per_fold'])}"
    )
    print("-" * 68)
    print(
        f"  CV ACCURACY : {r['acc_mean']*100:.2f}% +/- {r['acc_std']*100:.2f}  "
        f"(95% CI [{r['acc_ci'][0]*100:.2f}, {r['acc_ci'][1]*100:.2f}])"
    )
    print(
        f"  CV MACRO-F1 : {r['f1_mean']*100:.2f}% +/- {r['f1_std']*100:.2f}  "
        f"(95% CI [{r['f1_ci'][0]*100:.2f}, {r['f1_ci'][1]*100:.2f}])"
    )
    print(f"  (computed in {r['elapsed_s']:.1f}s)")
    print("=" * 68)
    # Ready-to-paste row for the report table tab:cv
    print("  >> paste into report Table tab:cv:")
    print(
        f"     {name} & "
        f"{r['acc_mean']*100:.2f} $\\pm$ {r['acc_std']*100:.2f} & "
        f"{r['f1_mean']*100:.2f} $\\pm$ {r['f1_std']*100:.2f} \\\\"
    )


def write_csv(path: Path, rows: list[tuple[str, dict]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "model",
                "folds",
                "seed",
                "n_samples",
                "n_features",
                "n_classes",
                "acc_mean_pct",
                "acc_std_pct",
                "acc_ci_low_pct",
                "acc_ci_high_pct",
                "f1_mean_pct",
                "f1_std_pct",
                "f1_ci_low_pct",
                "f1_ci_high_pct",
                "acc_per_fold_pct",
                "f1_per_fold_pct",
            ]
        )
        for name, r in rows:
            w.writerow(
                [
                    name,
                    r["folds"],
                    r["seed"],
                    r["n_samples"],
                    r["n_features"],
                    r["n_classes"],
                    f"{r['acc_mean']*100:.4f}",
                    f"{r['acc_std']*100:.4f}",
                    f"{r['acc_ci'][0]*100:.4f}",
                    f"{r['acc_ci'][1]*100:.4f}",
                    f"{r['f1_mean']*100:.4f}",
                    f"{r['f1_std']*100:.4f}",
                    f"{r['f1_ci'][0]*100:.4f}",
                    f"{r['f1_ci'][1]*100:.4f}",
                    ";".join(f"{x*100:.4f}" for x in r["acc_per_fold"]),
                    ";".join(f"{x*100:.4f}" for x in r["f1_per_fold"]),
                ]
            )
    print(f"\n[ok] wrote {path}")


# ===========================================================================
# CLI
# ===========================================================================
_CACHE_SNIPPET = """
# --- add this near the end of train_ocr_model.py, after X (HOG) and y exist ---
# It caches the exact feature matrix so cross_validate_ocr.py is fast & exact.
import numpy as np, os
os.makedirs(args.output_dir, exist_ok=True)
np.savez_compressed(os.path.join(args.output_dir, "features.npz"), X=X, y=y)
# Optional: also dump a metadata.json so CV recovers the HOG/SVM config:
import json
with open(os.path.join(args.output_dir, "metadata.json"), "w") as fh:
    json.dump({
        "feature_method": args.feature,                 # e.g. "hog_legacy" -> HOG324
        "hog": {"image_size": [32, 32], "pixels_per_cell": [8, 8],
                 "cells_per_block": [2, 2], "orientations": 9,
                 "block_norm": "L2-Hys", "binarize": False},
        "svm": {"C": 10.0, "kernel": "rbf", "gamma": "scale"},
    }, fh, indent=2)
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stratified k-fold CV for the HOG + RBF-SVM OCR model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model-dir",
        action="append",
        default=[],
        help="Trained model directory (recovers HOG/SVM config). "
        "Repeatable to compare several models.",
    )
    p.add_argument(
        "--features",
        action="append",
        default=[],
        help="Cached features (.npz with X,y, or .npy / dir). "
        "Pair positionally with each --model-dir. Mode A.",
    )
    p.add_argument(
        "--data-dir",
        action="append",
        default=[],
        help="Class-folder image dataset to rebuild HOG from. "
        "Pair positionally with each --model-dir. Mode B.",
    )
    p.add_argument(
        "--name",
        action="append",
        default=[],
        help="Display name for each model (optional).",
    )
    p.add_argument("--folds", type=int, default=5, help="Number of CV folds.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--pixels-per-cell",
        type=int,
        default=None,
        help="Override HOG cell size (8 -> HOG324, 4 -> HOG1764).",
    )
    p.add_argument(
        "--binarize",
        dest="binarize",
        action="store_true",
        default=None,
        help="Otsu-binarize crops before HOG " "(only if training did).",
    )
    p.add_argument("--output", type=str, default=None, help="CSV output path.")
    p.add_argument(
        "--help-cache",
        action="store_true",
        help="Print a snippet to cache features in train_ocr_model.py " "and exit.",
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.help_cache:
        print(_CACHE_SNIPPET)
        return 0

    # Resolve a list of jobs. At least one data source is required.
    n_models = max(len(args.model_dir), len(args.features), len(args.data_dir), 1)

    def pick(lst, i):
        return lst[i] if i < len(lst) else None

    if not args.features and not args.data_dir:
        print("[fatal] provide data via --features (Mode A) or --data-dir (Mode B).")
        print(
            "        Run with --help-cache to see how to cache features at train time."
        )
        return 2

    rows: list[tuple[str, dict]] = []
    for i in range(n_models):
        mdir = pick(args.model_dir, i)
        feat = pick(args.features, i)
        ddir = pick(args.data_dir, i)
        nm = pick(args.name, i)

        model_dir = Path(mdir) if mdir else None
        name = nm or (model_dir.name if model_dir else f"model_{i+1}")

        hog, svm = recover_config(model_dir, args.pixels_per_cell, args.binarize)
        print(
            f"\n[config] {name}: HOG cells={hog.pixels_per_cell} "
            f"blocks={hog.cells_per_block} orient={hog.orientations} "
            f"-> expected dim {hog.expected_dim}; "
            f"SVM C={svm.C} kernel={svm.kernel} gamma={svm.gamma}"
        )

        # Load data
        if feat:
            X, y = load_cached_features(Path(feat))
            print(f"[data] cached features: X={X.shape} y={y.shape}")
            if X.shape[1] != hog.expected_dim:
                print(
                    f"[warn] feature dim {X.shape[1]} != expected {hog.expected_dim}; "
                    f"using cached features as-is (this is fine if training used "
                    f"this exact matrix)."
                )
        elif ddir:
            X, y = build_features_from_images(Path(ddir), hog)
            if X.shape[1] != hog.expected_dim:
                print(
                    f"[warn] extracted dim {X.shape[1]} != expected "
                    f"{hog.expected_dim}; check --pixels-per-cell / image size."
                )
        else:
            print(f"[skip] {name}: no --features or --data-dir provided.")
            continue

        # Sanity: stratification needs >= folds samples per class
        classes, counts = np.unique(y, return_counts=True)
        if counts.min() < args.folds:
            print(
                f"[warn] smallest class has {counts.min()} samples < folds="
                f"{args.folds}; reducing folds to {int(counts.min())}."
            )
            folds = max(2, int(counts.min()))
        else:
            folds = args.folds

        r = run_cv(X, y, svm, folds=folds, seed=args.seed)
        print_result(name, r)
        rows.append((name, r))

    if args.output and rows:
        write_csv(Path(args.output), rows)

    if not rows:
        print("[fatal] nothing was evaluated.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
