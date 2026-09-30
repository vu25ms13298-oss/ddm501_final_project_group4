# ==========================================
# Member 3: Section 7 — Character Classification (SVM)
# ==========================================
# This module implements dataset generation for synthetic characters,
# training of the SVM classifier on ResNet18 features, and saving of models.

import os
import random
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

from .segmentation import resize_pad_char

# Bảng ký tự cần nhận dạng (Loại bỏ I, O, Q theo quy định biển Việt Nam)
CHAR_CLASSES = "0123456789ABCDEFGHKLMNPRSTUVXYZ"


def _available_fonts():
    fonts = []
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/UbuntuSans[wdth,wght].ttf",
        "/usr/share/fonts/truetype/ubuntu/UbuntuSansMono[wght].ttf",
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono[wght].ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "C:\\Windows\\Fonts\\tahoma.ttf",
        "C:\\Windows\\Fonts\\consolab.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            for size in [40, 44, 48, 52, 56, 60, 66, 72, 78]:
                try:
                    fonts.append(ImageFont.truetype(fp, size))
                except Exception:
                    pass
    if not fonts:
        fonts = [ImageFont.load_default()]
    return fonts


def _degrade_char_image(arr):
    """
    Apply plate-like degradation while keeping white glyphs on black background.
    """
    if random.random() < 0.35:
        kernel_size = random.choice([2, 2, 3])
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        op = cv2.dilate if random.random() < 0.55 else cv2.erode
        arr = op(arr, kernel, iterations=1)

    if random.random() < 0.45:
        k = random.choice([3, 3, 5])
        arr = cv2.GaussianBlur(arr, (k, k), random.uniform(0.0, 0.8))

    if random.random() < 0.35:
        noise = np.random.normal(0, random.uniform(5, 22), arr.shape).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if random.random() < 0.35:
        _, arr = cv2.threshold(arr, random.randint(35, 95), 255, cv2.THRESH_BINARY)

    if random.random() < 0.25:
        dropout = np.random.random(arr.shape) < random.uniform(0.005, 0.025)
        arr[dropout] = 0

    return arr


def generate_plate_style_synthetic_chars(
    char_classes=CHAR_CLASSES,
    samples_per_class=160,
    img_size=96,
):
    """
    Generate character crops that better resemble segmented license-plate glyphs.

    The original synthetic set is intentionally clean. This variant creates tall
    plate-like glyphs with rotation, shear, blur, stroke changes, thresholding,
    clipping, and noise so HOG+SVM sees samples closer to real debug crops.
    """
    fonts = _available_fonts()
    images, labels = [], []

    for cls_idx, ch in enumerate(char_classes):
        for _ in range(samples_per_class):
            font = random.choice(fonts)
            img = Image.new("L", (img_size, img_size), 0)
            draw = ImageDraw.Draw(img)
            bbox = draw.textbbox((0, 0), ch, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            x = (img_size - tw) // 2 - bbox[0] + random.randint(-5, 5)
            y = (img_size - th) // 2 - bbox[1] + random.randint(-5, 5)
            draw.text((x, y), ch, fill=255, font=font)
            arr = np.array(img)

            scale_x = random.uniform(0.68, 1.16)
            scale_y = random.uniform(0.88, 1.22)
            shear_x = random.uniform(-0.16, 0.16)
            angle = random.uniform(-10, 10)
            center = (img_size / 2, img_size / 2)
            rot = cv2.getRotationMatrix2D(center, angle, 1.0)
            affine = np.array(
                [
                    [scale_x, shear_x, (1 - scale_x) * center[0]],
                    [0.0, scale_y, (1 - scale_y) * center[1]],
                ],
                dtype=np.float32,
            )
            affine = np.vstack([affine, [0, 0, 1]]) @ np.vstack([rot, [0, 0, 1]])
            arr = cv2.warpAffine(
                arr,
                affine[:2],
                (img_size, img_size),
                flags=cv2.INTER_CUBIC,
                borderValue=0,
            )

            arr = _degrade_char_image(arr)

            ys, xs = np.where(arr > 35)
            if len(xs) > 0 and len(ys) > 0:
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                if random.random() < 0.28:
                    x1 = min(x2, x1 + random.randint(0, 2))
                if random.random() < 0.28:
                    y1 = min(y2, y1 + random.randint(0, 2))
                if random.random() < 0.18:
                    x2 = max(x1, x2 - random.randint(0, 1))
                cropped = arr[y1 : y2 + 1, x1 : x2 + 1]
                final = resize_pad_char(
                    cropped, size=32, pad_ratio=random.uniform(0.08, 0.18)
                )
            else:
                final = np.zeros((32, 32), dtype=np.uint8)

            images.append(final)
            labels.append(cls_idx)

    return np.array(images), np.array(labels)


def generate_synthetic_chars(
    char_classes=CHAR_CLASSES, samples_per_class=80, img_size=64
):
    """
    Sinh dataset ký tự synthetic với augmentation.

    Args:
        char_classes: chuỗi ký tự các class cần sinh
        samples_per_class: số lượng mẫu trên mỗi class
        img_size: kích thước ảnh render gốc (trước khi crop)

    Returns:
        tuple: (X_imgs, y_labels)
               - X_imgs: np.array (N, 32, 32) uint8 (các ảnh nhị phân đã crop & pad)
               - y_labels: np.array (N,) các nhãn class (index)
    """
    fonts = _available_fonts()

    images, labels = [], []

    for cls_idx, ch in enumerate(char_classes):
        for _ in range(samples_per_class):
            font = random.choice(fonts)
            # Render
            img = Image.new("L", (img_size, img_size), 0)
            draw = ImageDraw.Draw(img)
            bbox = draw.textbbox((0, 0), ch, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            x = (img_size - tw) // 2 - bbox[0] + random.randint(-3, 3)
            y = (img_size - th) // 2 - bbox[1] + random.randint(-3, 3)
            draw.text((x, y), ch, fill=255, font=font)
            arr = np.array(img)

            # Augmentation
            # 1. Slight rotation
            angle = random.uniform(-12, 12)
            M = cv2.getRotationMatrix2D((img_size / 2, img_size / 2), angle, 1.0)
            arr = cv2.warpAffine(arr, M, (img_size, img_size), borderValue=0)

            # 2. Random noise
            if random.random() < 0.5:
                noise = np.random.randint(0, 30, arr.shape, dtype=np.uint8)
                arr = cv2.add(arr, noise)

            # 3. Slight blur
            if random.random() < 0.3:
                arr = cv2.GaussianBlur(arr, (3, 3), 0)

            # Find bounding box & crop tight
            ys, xs = np.where(arr > 50)
            if len(xs) > 0 and len(ys) > 0:
                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()
                cropped = arr[y1 : y2 + 1, x1 : x2 + 1]
                # Resize/pad to 32x32 (same as segmentation output)
                final = resize_pad_char(cropped, size=32)
            else:
                final = np.zeros((32, 32), dtype=np.uint8)

            images.append(final)
            labels.append(cls_idx)

    return np.array(images), np.array(labels)


def train_svm(X_train, y_train, seed=42):
    """
    Train SVM (RBF kernel) classifier on scaled features.

    Args:
        X_train: features training set (N, 512)
        y_train: labels training set (N,)
        seed: random state seed

    Returns:
        tuple: (trained_svm_model, fitted_scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    svm_model = SVC(kernel="rbf", C=10, gamma="scale", random_state=seed)
    svm_model.fit(X_train_scaled, y_train)

    return svm_model, scaler


def save_models(
    svm_model,
    scaler,
    save_dir,
    feature_method="resnet",
    classifier_name="svm",
    metrics=None,
):
    """
    Lưu models để Member 4 sử dụng tích hợp.

    Args:
        svm_model: trained OCR classifier
        scaler: fitted StandardScaler
        save_dir: directory path to save model files
        feature_method: feature extractor used before the classifier
        classifier_name: classifier family/name
        metrics: optional evaluation metrics to persist
    """
    os.makedirs(save_dir, exist_ok=True)

    joblib.dump(svm_model, os.path.join(save_dir, "classifier.joblib"))
    joblib.dump(scaler, os.path.join(save_dir, "scaler.joblib"))

    # Backward-compatible filenames used by the original notebook/pipeline.
    joblib.dump(svm_model, os.path.join(save_dir, "svm_classifier.pkl"))
    joblib.dump(scaler, os.path.join(save_dir, "feature_scaler.pkl"))

    metadata = {
        "char_classes": list(CHAR_CLASSES),
        "char_size": 32,
        "feature_method": feature_method,
        "classifier": classifier_name,
        "metrics": metrics or {},
    }

    with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Backward-compatible class mapping file.
    with open(os.path.join(save_dir, "char_classes.json"), "w", encoding="utf-8") as f:
        json.dump({"classes": list(CHAR_CLASSES)}, f, indent=2, ensure_ascii=False)

    print(f"✅ Đã lưu models tại {save_dir}")
