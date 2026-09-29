# ==========================================
# Member 2: Image Preprocessing
# ==========================================
# Full preprocessing pipeline:
#   1. CLAHE contrast enhancement (per-channel in LAB space)
#   2. Bilateral filter denoising (edge-preserving)
#   3. Deskew via MinAreaRect angle estimation
#   4. 180° rotational ambiguity resolution (max valid char count)

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Scene-level preprocessing
# ---------------------------------------------------------------------------

def preprocess_scene_image(image):
    """
    Preprocess raw scene image before plate detection.

    Steps:
        - CLAHE on L channel (LAB space) for contrast enhancement
        - Bilateral filter for edge-preserving denoising

    Args:
        image: np.array (RGB scene image, uint8)

    Returns:
        np.array: preprocessed RGB image (same shape)
    """
    h, w = image.shape[:2]
    if max(h, w) > 1280:
        scale = 1280.0 / max(h, w)
        image = cv2.resize(
            image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    # CLAHE on luminance channel
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    # Bilateral filter – denoise while keeping plate edges sharp
    denoised = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

    # Light gamma correction for dark street images.
    gamma = 1.1
    table = np.array(
        [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(denoised, table)


def enhance_plate_crop_for_ocr(plate_img):
    """
    Enhance an already-cropped plate before binarization/OCR.

    The operation is deliberately local to the crop: CLAHE raises low-contrast
    characters, bilateral filtering removes compression noise, and unsharp
    masking makes thin strokes easier for HOG + SVM.
    """
    if plate_img is None or plate_img.size == 0:
        return plate_img

    is_color = len(plate_img.shape) == 3
    rgb = plate_img if is_color else cv2.cvtColor(plate_img, cv2.COLOR_GRAY2RGB)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=45, sigmaSpace=45)

    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.1)
    sharpened = cv2.addWeighted(enhanced, 1.65, blurred, -0.65, 0)
    return sharpened if is_color else cv2.cvtColor(sharpened, cv2.COLOR_RGB2GRAY)


# ---------------------------------------------------------------------------
# Plate-level deskew
# ---------------------------------------------------------------------------

def rotate_bound(image, angle, border_mode=cv2.BORDER_REPLICATE):
    """
    Rotate an image without clipping its corners.
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=border_mode,
    )


def estimate_plate_skew_angle(plate_img):
    """
    Estimate the dominant text/plate horizontal angle in degrees.

    Negative means the long text lines slope downward from left to right in
    image coordinates. The returned value is meant to be passed directly to
    cv2.getRotationMatrix2D / rotate_bound.
    """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_RGB2GRAY) if len(plate_img.shape) == 3 else plate_img.copy()
    h, w = gray.shape[:2]
    if h < 20 or w < 40:
        return 0.0

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 150)
    min_len = max(20, int(w * 0.22))
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(20, int(w * 0.08)),
        minLineLength=min_len,
        maxLineGap=max(8, int(w * 0.04)),
    )

    angles = []
    weights = []
    if lines is not None:
        for line in lines[:, 0]:
            x1, y1, x2, y2 = line
            dx = x2 - x1
            dy = y2 - y1
            length = float(np.hypot(dx, dy))
            if length < min_len:
                continue
            angle = float(np.degrees(np.arctan2(dy, dx)))
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90
            if abs(angle) <= 35:
                angles.append(angle)
                weights.append(length)

    if angles:
        order = np.argsort(angles)
        sorted_angles = np.array(angles)[order]
        sorted_weights = np.array(weights)[order]
        cutoff = sorted_weights.sum() / 2
        return float(sorted_angles[np.searchsorted(np.cumsum(sorted_weights), cutoff)])

    # Fallback: estimate from bright plate body. This handles cases where the
    # border is visible but Hough did not return clean long lines.
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 20:
        return 0.0
    rect = cv2.minAreaRect(coords)
    angle = float(rect[-1])
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    # minAreaRect uses the opposite sign convention from image-line angle in
    # this fallback branch.
    return -angle if abs(angle) <= 35 else 0.0


def crop_plate_body_after_deskew(plate_img, margin=0.04):
    """
    Crop away the padded scene background created by rotate_bound.

    This uses the bright plate body contour. If no reliable plate-like contour
    is found, the original image is returned unchanged.
    """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_RGB2GRAY) if len(plate_img.shape) == 3 else plate_img.copy()
    h, w = gray.shape[:2]
    if h < 20 or w < 40:
        return plate_img

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    image_area = h * w

    def find_best(binary, min_area_ratio, ar_range=(0.75, 6.5), penalize_border=True):
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_box = None
        best_score = 0.0
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if area < image_area * min_area_ratio:
                continue
            ar = bw / float(max(1, bh))
            if not ar_range[0] <= ar <= ar_range[1]:
                continue
            touches_border = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
            score = area
            if penalize_border and touches_border:
                score *= 0.35
            if score > best_score:
                best_score = score
                best_box = (x, y, bw, bh)
        return best_box

    # Prefer the bright physical plate body. This removes car/background bands
    # around a white/yellow plate that otherwise break aspect-ratio detection.
    best = None
    for thresh in (210, 190, 170, max(150, int(np.percentile(blurred, 72)))):
        _, bright = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5))
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel, iterations=2)
        best = find_best(bright, min_area_ratio=0.08, ar_range=(1.1, 6.8))
        if best is not None:
            break

    if best is None:
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        best = find_best(binary, min_area_ratio=0.06, ar_range=(0.75, 6.5))

    if best is None:
        return plate_img

    x, y, bw, bh = best
    mx = int(bw * margin)
    my = int(bh * margin)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w, x + bw + mx)
    y2 = min(h, y + bh + my)
    if x2 - x1 < 40 or y2 - y1 < 20:
        return plate_img
    return plate_img[y1:y2, x1:x2]

def _order_quad_points(points):
    pts = points.reshape(4, 2).astype("float32")
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).ravel()
    return np.array(
        [
            pts[np.argmin(sums)],
            pts[np.argmin(diffs)],
            pts[np.argmax(sums)],
            pts[np.argmax(diffs)],
        ],
        dtype="float32",
    )


def rectify_plate_crop(plate_img):
    """
    Perspective-warp the plate inside a YOLO bbox when a reliable quadrilateral
    border is visible. Falls back to the original crop if detection is unsure.
    """
    is_color = len(plate_img.shape) == 3
    gray = cv2.cvtColor(plate_img, cv2.COLOR_RGB2GRAY) if is_color else plate_img.copy()
    h, w = gray.shape[:2]
    if h < 20 or w < 40:
        return plate_img

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 140)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crop_area = h * w
    best_quad = None
    best_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < crop_area * 0.18:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
        if len(approx) != 4:
            continue
        x, y, bw, bh = cv2.boundingRect(approx)
        ar = bw / float(max(1, bh))
        if not 1.2 <= ar <= 6.5:
            continue
        if area > best_area:
            best_area = area
            best_quad = approx

    if best_quad is None:
        return plate_img

    rect = _order_quad_points(best_quad)
    tl, tr, br, bl = rect
    dst_w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    dst_h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if dst_w < 40 or dst_h < 15:
        return plate_img

    dst = np.array(
        [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(
        plate_img,
        matrix,
        (dst_w, dst_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def deskew_plate(plate_img):
    """
    Correct the rotation of a cropped plate image using MinAreaRect on binary
    foreground pixels.  Handles angles in the range [-45°, 45°].

    Args:
        plate_img: np.array – RGB (H,W,3) or grayscale (H,W) plate crop, uint8

    Returns:
        np.array: deskewed plate (same dtype/shape as input)
    """
    deskewed, _ = deskew_plate_with_angle(plate_img)
    return deskewed


def deskew_plate_with_angle(plate_img):
    """
    Deskew and return the applied rotation angle in degrees.
    """
    angle = estimate_plate_skew_angle(plate_img)
    if abs(angle) < 1.0:
        return plate_img, 0.0
    rotated = rotate_bound(plate_img, angle)
    return crop_plate_body_after_deskew(rotated), angle


def _count_valid_chars(plate_img):
    """
    Count the number of character-like connected components in a plate image.
    Used internally to resolve 180° rotational ambiguity.

    Returns:
        int: number of valid character candidates
    """
    try:
        # Prefer the proper segmentation module when available
        from .segmentation import segment_plate
        from .plate_classifier import classify_plate_type

        plate_type, _ = classify_plate_type(plate_img)
        char_imgs, _, _ = segment_plate(plate_img, plate_type)
        return len(char_imgs)
    except Exception:
        pass

    # Lightweight fallback using connected components
    gray = (
        cv2.cvtColor(plate_img, cv2.COLOR_RGB2GRAY)
        if len(plate_img.shape) == 3
        else plate_img
    )
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    h, w = binary.shape
    count = 0
    for i in range(1, num_labels):
        _, _, cw, ch, area = stats[i]
        if w * 0.02 < cw < w * 0.35 and h * 0.3 < ch and area > 30:
            count += 1
    return count


def deskew_with_fallback(plate_img, svm_model=None, scaler=None):
    """
    Deskew a plate crop without resolving 180-degree orientation.

    Strategy
    --------
    Apply deskew_plate() to correct small tilt. The integrated pipeline resolves
    180-degree orientation later by comparing OCR/format candidates.

    Args:
        plate_img:  np.array (RGB plate crop)
        svm_model:  kept for backward-compatible call sites
        scaler:     kept for backward-compatible call sites

    Returns:
        np.array: deskewed RGB plate
    """
    # Only correct small tilt here. The integrated pipeline evaluates both
    # 0-degree and 180-degree OCR candidates later, where plate-format scoring
    # has more context. Doing 180-degree selection here can flip a good crop
    # before OCR has enough evidence.
    return deskew_plate(plate_img)


def _two_line_char_counts(plate_img):
    """
    Count character candidates on the upper and lower rows of a two-line plate.

    Vietnamese two-line plates usually have the shorter province/series row on
    top and the longer numeric row at the bottom. This helper is used only as a
    lightweight orientation heuristic.
    """
    try:
        from .plate_classifier import classify_plate_type
        from .segmentation import preprocess_plate, segment_characters, split_two_lines

        plate_type, _ = classify_plate_type(plate_img)
        if plate_type != "2line":
            return None
        _, binary = preprocess_plate(plate_img)
        top, bottom = split_two_lines(binary)
        return len(segment_characters(top)), len(segment_characters(bottom))
    except Exception:
        return None


def orient_plate_vn(plate_img, return_was_rotated=False):
    """
    Resolve 180-degree orientation for common Vietnamese two-line plates.

    The heuristic prefers the orientation where the bottom row has more
    character candidates than the top row. One-line plates are returned as-is.
    """
    counts_0 = _two_line_char_counts(plate_img)
    if counts_0 is None:
        return (plate_img, False) if return_was_rotated else plate_img

    rotated = cv2.rotate(plate_img, cv2.ROTATE_180)
    counts_180 = _two_line_char_counts(rotated)
    if counts_180 is None:
        return (plate_img, False) if return_was_rotated else plate_img

    score_0 = counts_0[1] - counts_0[0]
    score_180 = counts_180[1] - counts_180[0]
    if score_180 > score_0:
        return (rotated, True) if return_was_rotated else rotated
    return (plate_img, False) if return_was_rotated else plate_img
