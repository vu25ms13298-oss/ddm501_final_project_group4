#!/usr/bin/env python3
"""Train and save a character OCR model for the LPR pipeline.

Default model: HOG features + SVM classifier. This keeps the OCR branch focused
on classical Machine Learning while still allowing other feature/classifier
choices for experiments.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import (
    CHAR_CLASSES,
    generate_plate_style_synthetic_chars,
    generate_synthetic_chars,
    save_models,
)
from src.features import (
    extract_features as extract_resnet_features,
    extract_hog_features,
    extract_hog_legacy_features,
    extract_raw_features,
    extract_wavelet_features,
)


FEATURE_EXTRACTORS = {
    "hog": extract_hog_features,
    "hog_legacy": extract_hog_legacy_features,
    "wavelet": extract_wavelet_features,
    "raw": extract_raw_features,
    "resnet": extract_resnet_features,
}


def build_classifier(name: str, seed: int):
    if name == "svm":
        return SVC(C=10, kernel="rbf", gamma="scale", random_state=seed)
    if name == "logistic":
        return LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="saga",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=seed)
    raise ValueError(f"Unsupported classifier: {name}")


def collect_emnist(root: Path, char_list: list[str], samples_per_class: int):
    try:
        from torchvision.datasets import EMNIST
        import torchvision.transforms as T
    except Exception as exc:
        raise RuntimeError(f"torchvision EMNIST is unavailable: {exc}") from exc

    def collect_split(split: str, label_to_char, target_chars: set[str]):
        dataset = EMNIST(
            root=str(root),
            split=split,
            train=True,
            download=True,
            transform=T.ToTensor(),
        )
        buckets: dict[str, list[np.ndarray]] = defaultdict(list)
        pbar = tqdm(dataset, desc=f"EMNIST {split}", unit="img")
        for img_t, label in pbar:
            ch = label_to_char(int(label))
            if ch is None or ch not in target_chars:
                continue
            if len(buckets[ch]) >= samples_per_class:
                continue
            arr = (img_t.squeeze().numpy().T * 255).astype(np.uint8)
            arr = cv2.resize(arr, (32, 32), interpolation=cv2.INTER_AREA)
            buckets[ch].append(arr)
            if all(len(buckets[ch]) >= samples_per_class for ch in target_chars):
                break

        xs, ys = [], []
        for ch in sorted(target_chars):
            imgs = buckets.get(ch, [])
            xs.extend(imgs)
            ys.extend([char_list.index(ch)] * len(imgs))
        return xs, ys

    digit_chars = {ch for ch in char_list if ch.isdigit()}
    letter_chars = {ch for ch in char_list if ch.isalpha()}

    xs_d, ys_d = collect_split("digits", lambda label: str(label), digit_chars)

    # EMNIST letters are 1-indexed: A=1, B=2, ...
    # Vietnamese plate OCR classes used here exclude I, J, O, Q, W.
    skip_labels = {9, 10, 15, 17, 23}

    def letter_label_to_char(label: int):
        if label in skip_labels:
            return None
        return chr(64 + label)

    xs_l, ys_l = collect_split("letters", letter_label_to_char, letter_chars)
    xs = xs_d + xs_l
    ys = ys_d + ys_l
    if not xs:
        return np.empty((0, 32, 32), dtype=np.uint8), np.empty(0, dtype=int)
    return np.array(xs, dtype=np.uint8), np.array(ys, dtype=int)


def augment_real_char(img: np.ndarray, copies: int):
    images = []
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    for _ in range(copies):
        arr = img.copy()
        angle = random.uniform(-4, 4)
        scale = random.uniform(0.92, 1.08)
        matrix = cv2.getRotationMatrix2D((16, 16), angle, scale)
        matrix[0, 2] += random.uniform(-1.2, 1.2)
        matrix[1, 2] += random.uniform(-1.2, 1.2)
        arr = cv2.warpAffine(
            arr, matrix, (32, 32), flags=cv2.INTER_CUBIC, borderValue=0
        )
        if random.random() < 0.35:
            arr = cv2.GaussianBlur(arr, (3, 3), random.uniform(0.0, 0.6))
        if random.random() < 0.3:
            noise = np.random.normal(0, random.uniform(3, 12), arr.shape).astype(
                np.int16
            )
            arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        if random.random() < 0.25:
            _, arr = cv2.threshold(arr, random.randint(35, 90), 255, cv2.THRESH_BINARY)
        images.append(arr)
    return images


def collect_labeled_char_dir(root: Path, char_list: list[str], augmentations: int):
    if root is None or not root.exists():
        return np.empty((0, 32, 32), dtype=np.uint8), np.empty(0, dtype=int)

    xs, ys = [], []
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in image_exts:
            continue
        label = None
        if path.parent.name in char_list:
            label = path.parent.name
        elif path.stem and path.stem[0].upper() in char_list:
            label = path.stem[0].upper()
        if label is None:
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        samples = [cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)]
        samples.extend(augment_real_char(img, augmentations))
        xs.extend(samples)
        ys.extend([char_list.index(label)] * len(samples))

    if not xs:
        return np.empty((0, 32, 32), dtype=np.uint8), np.empty(0, dtype=int)
    return np.array(xs, dtype=np.uint8), np.array(ys, dtype=int)


def collect_debug_chars(
    debug_dir: Path, label: str, char_list: list[str], augmentations: int
):
    if not debug_dir or not label:
        return np.empty((0, 32, 32), dtype=np.uint8), np.empty(0, dtype=int)
    label = "".join(ch for ch in label.upper() if ch in char_list)
    paths = sorted(debug_dir.glob("char_*.png"))
    if len(paths) != len(label):
        print(
            f"Skipping debug chars: {len(paths)} crops but label has {len(label)} chars"
        )
        return np.empty((0, 32, 32), dtype=np.uint8), np.empty(0, dtype=int)

    xs, ys = [], []
    for path, ch in zip(paths, label):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        samples = [cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)]
        samples.extend(augment_real_char(img, augmentations))
        xs.extend(samples)
        ys.extend([char_list.index(ch)] * len(samples))

    if not xs:
        return np.empty((0, 32, 32), dtype=np.uint8), np.empty(0, dtype=int)
    return np.array(xs, dtype=np.uint8), np.array(ys, dtype=int)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train OCR character classifier for LPR."
    )
    parser.add_argument("--feature", choices=FEATURE_EXTRACTORS.keys(), default="hog")
    parser.add_argument(
        "--classifier",
        choices=["svm", "logistic", "knn", "random_forest"],
        default="svm",
    )
    parser.add_argument("--samples-per-class", type=int, default=250)
    parser.add_argument("--plate-style-samples-per-class", type=int, default=400)
    parser.add_argument("--emnist-per-class", type=int, default=200)
    parser.add_argument("--no-emnist", action="store_true")
    parser.add_argument(
        "--labeled-char-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "characters" / "labeled",
    )
    parser.add_argument("--real-augmentations", type=int, default=60)
    parser.add_argument("--debug-char-dir", type=Path, default=None)
    parser.add_argument("--debug-label", type=str, default="")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "ocr_hog_svm",
    )
    parser.add_argument(
        "--emnist-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "characters" / "emnist",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    char_list = list(CHAR_CLASSES)
    print(f"Classes ({len(char_list)}): {''.join(char_list)}")
    print(
        f"Training OCR model: feature={args.feature}, " f"classifier={args.classifier}"
    )

    print(f"Generating synthetic characters: {args.samples_per_class}/class")
    x_syn, y_syn = generate_synthetic_chars(
        char_classes=CHAR_CLASSES,
        samples_per_class=args.samples_per_class,
        img_size=64,
    )
    print(f"Synthetic dataset: {x_syn.shape}")

    datasets_x = [x_syn]
    datasets_y = [y_syn]

    if args.plate_style_samples_per_class > 0:
        print(
            "Generating plate-style synthetic characters: "
            f"{args.plate_style_samples_per_class}/class"
        )
        x_plate, y_plate = generate_plate_style_synthetic_chars(
            char_classes=CHAR_CLASSES,
            samples_per_class=args.plate_style_samples_per_class,
            img_size=96,
        )
        print(f"Plate-style synthetic dataset: {x_plate.shape}")
        datasets_x.append(x_plate)
        datasets_y.append(y_plate)

    if not args.no_emnist:
        print(f"Loading EMNIST characters: {args.emnist_per_class}/class")
        x_em, y_em = collect_emnist(args.emnist_root, char_list, args.emnist_per_class)
        print(f"EMNIST dataset: {x_em.shape}")
        if len(x_em):
            datasets_x.append(x_em)
            datasets_y.append(y_em)

    x_labeled, y_labeled = collect_labeled_char_dir(
        args.labeled_char_dir,
        char_list,
        args.real_augmentations,
    )
    if len(x_labeled):
        print(f"Labeled real char dataset: {x_labeled.shape}")
        datasets_x.append(x_labeled)
        datasets_y.append(y_labeled)

    x_debug, y_debug = collect_debug_chars(
        args.debug_char_dir,
        args.debug_label,
        char_list,
        args.real_augmentations,
    )
    if len(x_debug):
        print(f"Debug-labeled char dataset: {x_debug.shape}")
        datasets_x.append(x_debug)
        datasets_y.append(y_debug)

    x_all = np.concatenate(datasets_x, axis=0)
    y_all = np.concatenate(datasets_y, axis=0)
    print(f"Total OCR samples: {len(x_all)}")

    x_train, x_test, y_train, y_test = train_test_split(
        x_all,
        y_all,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_all,
    )
    print(f"Train/test split: {len(x_train)} / {len(x_test)}")

    feature_fn = FEATURE_EXTRACTORS[args.feature]
    print("Extracting train features...")
    x_train_f = feature_fn(list(x_train))
    print("Extracting test features...")
    x_test_f = feature_fn(list(x_test))
    print(f"Feature shapes: train={x_train_f.shape}, test={x_test_f.shape}")

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train_f)
    x_test_s = scaler.transform(x_test_f)

    clf = build_classifier(args.classifier, args.seed)
    print("Fitting classifier...")
    clf.fit(x_train_s, y_train)

    y_pred = clf.predict(x_test_s)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "synthetic_per_class": int(args.samples_per_class),
        "plate_style_synthetic_per_class": int(args.plate_style_samples_per_class),
        "emnist_per_class": 0 if args.no_emnist else int(args.emnist_per_class),
    }
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1 : {metrics['macro_f1']:.4f}")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=list(range(len(char_list))),
            target_names=char_list,
            zero_division=0,
        )
    )

    save_models(
        clf,
        scaler,
        args.output_dir,
        feature_method=args.feature,
        classifier_name=args.classifier,
        metrics=metrics,
        feature_dim=x_train_f.shape[1],
    )
    print(f"Saved OCR model to: {args.output_dir}")


if __name__ == "__main__":
    main()
