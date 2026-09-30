# ==========================================
# Member 3: Section 6 — Feature Extraction (ResNet18)
# ==========================================
# This module implements feature extraction using a pre-trained ResNet18 model.
# Image 128x128 (or 32x32 resized to 224x224) -> ResNet18 -> feature vector 1x512.

import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2


class ResNet18FeatureExtractor(nn.Module):
    """
    Wrapper cho ResNet18 pre-trained: bỏ FC layer cuối, output 512-d feature.
    """

    def __init__(self):
        super().__init__()
        # Load pre-trained ResNet18
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Bỏ avgpool và fc — ta giữ lại tới layer trước fc
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # đến avgpool
        self.eval()

    def forward(self, x):
        with torch.no_grad():
            feat = self.features(x)  # [B, 512, 1, 1]
            feat = feat.view(feat.size(0), -1)  # [B, 512]
        return feat


# Global settings for device, transform and encoder initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform để đưa ảnh ký tự (grayscale 32x32) về dạng input của ResNet (RGB 224x224)
char_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # 1 channel → 3 channel
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Initialize global encoder instance, lazily or statically
_encoder = None


def get_encoder():
    """Lazily loads and returns the global ResNet18 feature extractor model."""
    global _encoder
    if _encoder is None:
        try:
            _encoder = ResNet18FeatureExtractor().to(device)
            print("✅ ResNet18 Feature Extractor loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading ResNet18 model: {e}")
            raise e
    return _encoder


def extract_features(char_imgs, batch_size=64, encoder_model=None):
    """
    Trích xuất feature từ list các ảnh ký tự.

    Args:
        char_imgs: list of np.array (HxW grayscale, uint8)
        batch_size: size of batches for feature extraction
        encoder_model: custom feature extractor model (optional)

    Returns:
        np.array shape (N, 512)
    """
    if len(char_imgs) == 0:
        return np.zeros((0, 512), dtype=np.float32)

    encoder = encoder_model if encoder_model is not None else get_encoder()

    features_list = []
    for i in range(0, len(char_imgs), batch_size):
        batch = char_imgs[i : i + batch_size]
        batch_tensor = torch.stack([char_transform(img) for img in batch]).to(device)
        feats = encoder(batch_tensor)
        features_list.append(feats.cpu().numpy())
    return np.vstack(features_list)


# ===========================================================================
# Classical feature extractors (HOG, Haar Wavelet, Raw pixel)
# ===========================================================================


def extract_raw_features(char_imgs):
    """
    Raw pixel baseline: flatten each 32×32 char image → 1024-d vector, normalised [0,1].

    Args:
        char_imgs: list/array of np.array (H×W, uint8)

    Returns:
        np.array shape (N, 784)
    """
    out = []
    for img in char_imgs:
        # Ensure 32×32
        if img.shape != (32, 32):
            img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
        out.append(img.flatten().astype(np.float32) / 255.0)
    return np.array(out, dtype=np.float32)


def extract_hog_features(
    char_imgs, pixels_per_cell=(4, 4), cells_per_block=(2, 2), orientations=9
):
    """
    Histogram of Oriented Gradients (HOG) features for character images.
    Default config follows LPR_Report.pdf on 32×32: 1764-d HOG vector.

    Args:
        char_imgs:        list/array of np.array (H×W, uint8)
        pixels_per_cell:  HOG cell size in pixels
        cells_per_block:  number of cells per normalisation block
        orientations:     number of gradient orientation bins

    Returns:
        np.array shape (N, D) where D = 1764 for default params on 32×32
    """
    try:
        from skimage.feature import hog as skimage_hog
    except ImportError as exc:
        raise ImportError(
            "scikit-image is required for HOG: pip install scikit-image"
        ) from exc

    out = []
    for img in char_imgs:
        if img.shape != (32, 32):
            img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
        feat = skimage_hog(
            img.astype(np.float32) / 255.0,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm="L2-Hys",
            feature_vector=True,
        )
        out.append(feat)
    return np.array(out, dtype=np.float32)


def extract_hog_legacy_features(char_imgs):
    """
    Legacy HOG config used by older saved models: 32x32 with 8x8 cells -> 324-d.
    """
    return extract_hog_features(
        char_imgs,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        orientations=9,
    )


def extract_wavelet_features(char_imgs, wavelet="haar", level=2):
    """
    Haar Wavelet decomposition features for 32×32 images.

    The output vector is the concatenation of the flattened level-2 approximation
    sub-band and the three level-2 detail sub-bands.

    Args:
        char_imgs: list/array of np.array (H×W, uint8)
        wavelet:   PyWavelets wavelet name (default 'haar')
        level:     decomposition level (default 2)

    Returns:
        np.array shape (N, 196)
    """
    try:
        import pywt
    except ImportError as exc:
        raise ImportError("PyWavelets is required: pip install PyWavelets") from exc

    out = []
    for img in char_imgs:
        if img.shape != (32, 32):
            img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
        img_f = img.astype(np.float32) / 255.0
        coeffs = pywt.wavedec2(img_f, wavelet=wavelet, level=level)
        feat = list(coeffs[0].flatten())
        for sub in coeffs[1]:
            feat.extend(sub.flatten().tolist())
        out.append(feat)
    return np.array(out, dtype=np.float32)
