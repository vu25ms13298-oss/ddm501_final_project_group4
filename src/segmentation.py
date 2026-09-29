# ==========================================
# Member 3: Section 5 — Character Segmentation
# ==========================================
# This module is responsible for preprocessing the plate crop,
# splitting two-line plates, and segmenting/extracting characters.

import cv2
import numpy as np


def _to_gray(plate_img):
    if len(plate_img.shape) == 3:
        return cv2.cvtColor(plate_img, cv2.COLOR_RGB2GRAY)
    return plate_img.copy()


def _odd_at_least(value, minimum):
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


def _enhance_gray(gray):
    denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=45, sigmaSpace=45)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    contrast = clahe.apply(denoised)
    blurred = cv2.GaussianBlur(contrast, (0, 0), 1.1)
    return cv2.addWeighted(contrast, 1.75, blurred, -0.75, 0)


def _clean_binary(binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return cv2.medianBlur(binary, 3)


def _close_binary(binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def _binary_component_score(binary_img):
    h, w = binary_img.shape
    mx = int(w * 0.02)
    my = int(h * 0.08)
    inner = binary_img[my:h - my, mx:w - mx] if h - 2 * my > 5 and w - 2 * mx > 5 else binary_img
    H, W = inner.shape
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(inner, connectivity=8)

    count = 0
    area_score = 0.0
    huge_penalty = 0.0
    edge_penalty = 0.0
    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]
        component_area = cw * ch
        if area > H * W * 0.18 or (cw > W * 0.50 and ch > H * 0.35):
            huge_penalty += 8.0
            continue
        if area < max(12, H * W * 0.00045):
            continue
        if ch < H * 0.10 or ch > H * 0.82:
            continue
        if cw < W * 0.008 or cw > W * 0.30:
            continue
        aspect = cw / float(max(1, ch))
        if aspect > 1.9:
            continue
        if x <= 1 or x + cw >= W - 1:
            edge_penalty += 0.8
        extent = area / float(max(1, component_area))
        count += 1
        area_score += min(1.4, area / float(max(1, H * W * 0.025))) + min(0.4, extent)

    fill = float(np.count_nonzero(binary_img)) / binary_img.size
    if 0.035 <= fill <= 0.28:
        fill_score = 4.0
    elif fill <= 0.40:
        fill_score = 1.0
    else:
        fill_score = -10.0 * (fill - 0.40)

    count_score = count * 3.0
    if 6 <= count <= 12:
        count_score += 8.0 - abs(8 - min(count, 10))
    elif count == 0:
        count_score -= 12.0

    return count_score + area_score + fill_score - huge_penalty - edge_penalty


def preprocess_plate_candidates(plate_img):
    """
    Build several binary versions of a plate crop and rank them.

    White/yellow plates need dark-character extraction (binary inverse /
    black-hat). Blue/red plates usually need bright-character extraction
    (binary normal / top-hat). Trying both polarities avoids hard-coding one
    plate color family.
    """
    gray = _to_gray(plate_img)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    enhanced = _enhance_gray(gray)
    h, w = gray.shape[:2]

    raw_candidates = []
    for name, img in (("gray", gray), ("enhanced", enhanced)):
        for mode_name, mode in (("light", cv2.THRESH_BINARY), ("dark", cv2.THRESH_BINARY_INV)):
            _, binary = cv2.threshold(img, 0, 255, mode + cv2.THRESH_OTSU)
            raw_candidates.append((f"{name}_{mode_name}_otsu", binary))

    block = _odd_at_least(min(h, w) * 0.16, 21)
    block = min(block, _odd_at_least(min(h, w) - 2, 21)) if min(h, w) > 25 else 21
    for mode_name, mode in (("light", cv2.THRESH_BINARY), ("dark", cv2.THRESH_BINARY_INV)):
        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            mode,
            block,
            7,
        )
        raw_candidates.append((f"enhanced_{mode_name}_adaptive", binary))

    try:
        from skimage.filters import threshold_sauvola

        window = _odd_at_least(min(h, w) * 0.22, 25)
        if window < min(h, w):
            sauvola = threshold_sauvola(enhanced, window_size=window, k=0.18)
            raw_candidates.append((
                "enhanced_light_sauvola",
                ((enhanced > sauvola) * 255).astype(np.uint8),
            ))
            raw_candidates.append((
                "enhanced_dark_sauvola",
                ((enhanced < sauvola) * 255).astype(np.uint8),
            ))
    except Exception:
        pass

    for scale in (0.045, 0.065, 0.085):
        kx = max(15, int(w * scale))
        ky = max(7, int(h * scale * 0.55))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))

        blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
        _, binary = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        raw_candidates.append((f"blackhat_{scale:.3f}", binary))

        tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
        _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        raw_candidates.append((f"tophat_{scale:.3f}", binary))

    candidates = []
    seen = set()
    for name, binary in raw_candidates:
        binary = _clean_binary(binary)
        key = binary.tobytes()
        if key in seen:
            continue
        seen.add(key)
        candidates.append((name, binary, _binary_component_score(binary)))

    candidates.sort(key=lambda item: item[2], reverse=True)
    return gray, candidates


def preprocess_plate(plate_img):
    """
    Tiền xử lý ảnh biển số: grayscale + binarize.
    
    Args:
        plate_img: np.array (RGB or grayscale plate crop)
        
    Returns:
        tuple: (gray_image, binary_image)
    """
    gray, binary = preprocess_plate_legacy(plate_img)
    if len(_segment_binary_plate(binary, "2line")) < 4:
        gray, candidates = preprocess_plate_candidates(plate_img)
        binary = candidates[0][1] if candidates else np.zeros_like(gray)
    return gray, binary


def preprocess_plate_legacy(plate_img):
    """
    Original Otsu polarity heuristic. Kept as the primary path because it is
    stable on clean white plates; the richer candidate search is a fallback.
    """
    gray = _to_gray(plate_img)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh_val, binary_normal = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    _, binary_inv = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    dark_pixels = gray[gray < thresh_val]
    mean_dark = float(np.mean(dark_pixels)) if dark_pixels.size else 255.0
    heuristic_binary = binary_normal if mean_dark < 80 else binary_inv
    other_binary = binary_inv if heuristic_binary is binary_normal else binary_normal

    binary = (
        heuristic_binary
        if _binary_component_score(heuristic_binary) >= _binary_component_score(other_binary)
        else other_binary
    )
    flipped = cv2.bitwise_not(binary)
    fill = float(np.count_nonzero(binary)) / binary.size
    if fill > 0.60 and _binary_component_score(flipped) >= _binary_component_score(binary):
        binary = flipped

    return gray, _close_binary(binary)


def split_two_lines(binary):
    """
    Tách biển 2-dòng thành 2 phần dựa trên horizontal projection.
    
    Args:
        binary: np.array (binary plate image)
        
    Returns:
        tuple: (upper_line_image, lower_line_image)
    """
    h_proj = np.sum(binary > 0, axis=1).astype(np.float32)
    h_proj = cv2.GaussianBlur(h_proj.reshape(-1, 1), (1, 9), 0).ravel()
    h = binary.shape[0]
    y1 = max(1, int(h * 0.25))
    y2 = min(h - 1, int(h * 0.75))
    if y2 <= y1:
        valley = h // 2
    else:
        valley = y1 + int(np.argmin(h_proj[y1:y2]))
    return binary[:valley, :], binary[valley:, :]


def segment_characters(binary_line, char_min_width_ratio=0.02, char_max_width_ratio=0.3,
                       char_min_height_ratio=0.3):
    """
    Tách ký tự khỏi một dòng (binary image) bằng connected components.

    Args:
        binary_line: np.array (dòng ảnh nhị phân)
        char_min_width_ratio: tỉ lệ chiều rộng tối thiểu so với dòng
        char_max_width_ratio: tỉ lệ chiều rộng tối đa so với dòng
        char_min_height_ratio: tỉ lệ chiều cao tối thiểu so với dòng

    Returns:
        list of dict: [{"img": <ảnh ký tự>, "bbox": (x, y, w, h)}]
    """
    H, W = binary_line.shape
    # Connected components để có boxes chính xác hơn projection thuần
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_line, connectivity=8)

    chars = []
    for i in range(1, num_labels):  # Bỏ qua background (label 0)
        x, y, w, h, area = stats[i]
        # Lọc bằng kích thước tương đối
        if w < W * char_min_width_ratio or w > W * char_max_width_ratio:
            continue
        if h < H * char_min_height_ratio:
            continue
        if area < 30:
            continue
        char_img = binary_line[y:y + h, x:x + w]
        chars.append({"img": char_img, "bbox": (x, y, w, h)})

    # Sắp xếp theo tọa độ x (trái → phải)
    chars.sort(key=lambda c: c["bbox"][0])
    return chars


def segment_characters_by_projection(binary_line, min_width_ratio=0.012,
                                      max_width_ratio=0.18, min_height_ratio=0.28):
    """
    Split a one-line plate by vertical projection.

    This fallback is useful when dark glyphs touch the plate frame or each
    other, causing connected-components to merge several characters into one
    blob. Short punctuation-like groups are ignored.
    """
    H, W = binary_line.shape
    if H == 0 or W == 0:
        return []

    col_proj = np.sum(binary_line > 0, axis=0)
    active = col_proj > max(1, H * 0.05)

    groups = []
    start = None
    for idx, is_active in enumerate(active):
        if is_active and start is None:
            start = idx
        at_end = idx == W - 1
        if start is not None and ((not is_active) or at_end):
            end = idx if at_end and is_active else idx - 1
            if end >= start:
                groups.append((start, end))
            start = None

    chars = []
    for x1, x2 in groups:
        w = x2 - x1 + 1
        if w < W * min_width_ratio or w > W * max_width_ratio:
            continue
        patch = binary_line[:, x1:x2 + 1]
        ys, xs = np.where(patch > 0)
        if len(xs) == 0:
            continue
        y1, y2 = int(ys.min()), int(ys.max())
        x_local_1, x_local_2 = int(xs.min()), int(xs.max())
        h = y2 - y1 + 1
        tight_w = x_local_2 - x_local_1 + 1
        if h < H * min_height_ratio:
            continue
        if tight_w < W * min_width_ratio:
            continue
        x = x1 + x_local_1
        char_img = binary_line[y1:y2 + 1, x:x + tight_w]
        chars.append({"img": char_img, "bbox": (x, y1, tight_w, h)})

    chars.sort(key=lambda c: c["bbox"][0])
    return chars


def crop_inner_plate(binary, margin_x=0.08, margin_y=0.08):
    """
    Remove a small plate border before character segmentation.

    License plate frames often become foreground after thresholding and can be
    mistaken for characters. The margins are intentionally conservative.
    """
    h, w = binary.shape[:2]
    mx = min(int(w * margin_x), max(0, w // 4))
    my = min(int(h * margin_y), max(0, h // 4))
    if h - 2 * my <= 5 or w - 2 * mx <= 5:
        return binary
    return binary[my:h - my, mx:w - mx]


def component_char_candidates(binary_img):
    """
    Extract character-like connected components from an inner plate region.
    """
    H, W = binary_img.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)

    candidates = []
    row_area = max(1, H * W)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < max(30, row_area * 0.003):
            continue
        if h < H * 0.30 or h > H * 1.40:
            continue
        if w < W * 0.025 or w > W * 0.32:
            continue
        aspect = w / float(max(1, h))
        if aspect > 2.5:
            continue
        extent = area / float(max(1, w * h))
        if aspect > 0.75 and extent > 0.55 and h < H * 0.55:
            # Bolt holes / punctuation-like circular blobs.
            continue
        # Drop thin frame fragments on the left/right edge, but keep a real
        # character if the detector crop is tight and the glyph touches x=0.
        touches_edge = x <= 1 or x + w >= W - 1
        if touches_edge and (w < W * 0.05 or (h > H * 0.62 and aspect < 0.28)):
            continue

        char_img = binary_img[y:y + h, x:x + w]
        candidates.append({
            "img": char_img,
            "bbox": (x, y, w, h),
            "x_center": x + w / 2,
            "y_center": y + h / 2,
        })

    return candidates


def segment_two_line_components(binary_plate):
    """
    Segment two-line plates by clustering character candidates by vertical center.
    """
    inner = crop_inner_plate(binary_plate)
    top, bottom = split_two_lines(inner)
    top_chars = segment_line_components(top, y_offset=0, target_range=(3, 4))
    bottom_chars = segment_line_components(bottom, y_offset=top.shape[0], target_range=(4, 6))
    if len(top_chars) >= 2 and len(bottom_chars) >= 3:
        return top_chars + bottom_chars

    candidates = component_char_candidates(inner)
    if len(candidates) < 4:
        return top_chars + bottom_chars

    y_min = min(c["y_center"] for c in candidates)
    y_max = max(c["y_center"] for c in candidates)
    split_y = (y_min + y_max) / 2

    top_chars = [c for c in candidates if c["y_center"] < split_y]
    bottom_chars = [c for c in candidates if c["y_center"] >= split_y]
    if not top_chars or not bottom_chars:
        top, bottom = split_two_lines(inner)
        return segment_characters(top) + segment_characters(bottom)

    top_chars.sort(key=lambda c: c["bbox"][0])
    bottom_chars.sort(key=lambda c: c["bbox"][0])
    return top_chars + bottom_chars


def _is_edge_artifact(candidate, line_shape):
    H, W = line_shape
    x, y, w, h = candidate["bbox"]
    aspect = w / float(max(1, h))
    touches_left = x <= 1
    touches_right = x + w >= W - 1
    if (touches_left or touches_right) and h > H * 0.72 and w < W * 0.08:
        return True
    if (touches_left or touches_right) and aspect < 0.13:
        return True
    return False


def _filter_line_artifacts(chars, line_shape):
    H, W = line_shape
    heights = np.array([c["bbox"][3] for c in chars], dtype=np.float32)
    widths = np.array([c["bbox"][2] for c in chars], dtype=np.float32)
    median_h = max(1.0, float(np.median(heights))) if len(heights) else 1.0
    median_w = max(1.0, float(np.median(widths))) if len(widths) else 1.0
    filtered = []
    for c in chars:
        x, y, w, h = c["bbox"]
        aspect = w / float(max(1, h))
        touches_edge = x <= 1 or x + w >= W - 1
        if touches_edge and w < W * 0.045:
            continue
        if touches_edge and h > H * 0.72 and w < W * 0.08:
            continue
        if h < H * 0.18:
            continue
        if h < median_h * 0.40 and aspect > 0.75 and w < median_w * 0.90:
            continue
        if aspect > 1.9 and h < H * 0.55:
            continue
        filtered.append(c)
    return filtered


def _tight_candidate_from_span(binary_line, x1, x2, y1, y2):
    """
    Build a tight candidate inside a proposed x-span.

    This is used after a wide connected component is split by projection. The
    y-span starts from the original merged component, then the foreground pixels
    inside the half are tightened again to avoid carrying empty padding forward.
    """
    H, W = binary_line.shape
    x1 = max(0, min(W - 1, int(x1)))
    x2 = max(x1 + 1, min(W, int(x2)))
    y1 = max(0, min(H - 1, int(y1)))
    y2 = max(y1 + 1, min(H, int(y2)))
    patch = binary_line[y1:y2, x1:x2]
    ys, xs = np.where(patch > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    tx1 = x1 + int(xs.min())
    tx2 = x1 + int(xs.max()) + 1
    ty1 = y1 + int(ys.min())
    ty2 = y1 + int(ys.max()) + 1
    w = tx2 - tx1
    h = ty2 - ty1
    if w <= 1 or h <= 1:
        return None
    char_img = binary_line[ty1:ty2, tx1:tx2]
    return {
        "img": char_img,
        "bbox": (tx1, ty1, w, h),
        "x_center": tx1 + w / 2,
        "y_center": ty1 + h / 2,
    }


def _split_wide_candidate(binary_line, candidate, median_width):
    """
    Split a merged two-character component using the vertical projection valley.

    Connected-components is allowed to over-merge when two digits touch. For
    OCR, a clean but imperfect split is usually better than feeding a two-digit
    blob to a one-character SVM.
    """
    H, W = binary_line.shape
    x, y, w, h = candidate["bbox"]
    if median_width <= 0:
        return None

    aspect = w / float(max(1, h))
    too_wide = w >= max(median_width * 1.45, W * 0.09)
    if not too_wide or aspect < 0.62 or h < H * 0.28:
        return None

    patch = binary_line[y:y + h, x:x + w]
    if patch.size == 0:
        return None

    col_proj = np.sum(patch > 0, axis=0).astype(np.float32)
    if len(col_proj) < 8:
        return None

    kernel = np.ones(5, dtype=np.float32) / 5.0
    smooth = np.convolve(col_proj, kernel, mode="same")
    lo = max(2, int(w * 0.30))
    hi = min(w - 2, int(w * 0.70))
    if hi <= lo:
        return None

    split_local = lo + int(np.argmin(smooth[lo:hi]))
    left_peak = float(np.max(smooth[:split_local])) if split_local > 0 else 0.0
    right_peak = float(np.max(smooth[split_local + 1:])) if split_local + 1 < w else 0.0
    peak = max(1.0, min(left_peak, right_peak))
    valley_ratio = float(smooth[split_local]) / peak

    # If the valley is shallow, only split very wide components. This prevents
    # wide single letters such as M from being cut unnecessarily.
    if valley_ratio > 0.82 and w < median_width * 1.85:
        return None

    split_x = x + split_local
    left = _tight_candidate_from_span(binary_line, x, split_x + 1, y, y + h)
    right = _tight_candidate_from_span(binary_line, split_x + 1, x + w, y, y + h)
    if left is None or right is None:
        return None

    lw = left["bbox"][2]
    rw = right["bbox"][2]
    lh = left["bbox"][3]
    rh = right["bbox"][3]
    if lw < W * 0.012 or rw < W * 0.012:
        return None
    if lh < H * 0.18 or rh < H * 0.18:
        return None

    return [left, right]


def split_wide_char_candidates(binary_line, chars, target_max=9):
    """
    Split over-wide character candidates before OCR.

    The function is deliberately conservative: it uses the median width of the
    current line and only creates extra chars until target_max is reached.
    """
    if len(chars) < 2 or (target_max is not None and len(chars) >= target_max):
        return chars

    widths = np.array([c["bbox"][2] for c in chars], dtype=np.float32)
    median_width = float(np.median(widths)) if len(widths) else 0.0
    if median_width <= 0:
        return chars

    remaining_splits = None if target_max is None else max(0, target_max - len(chars))
    out = []
    for cand in sorted(chars, key=lambda c: c["bbox"][0]):
        if remaining_splits == 0:
            out.append(cand)
            continue
        split = _split_wide_candidate(binary_line, cand, median_width)
        if split:
            out.extend(split)
            if remaining_splits is not None:
                remaining_splits -= 1
        else:
            out.append(cand)

    out.sort(key=lambda c: c["bbox"][0])
    return out


def _score_line_chars(chars, line_shape, target_range=None):
    H, W = line_shape
    count = len(chars)
    if count == 0:
        return -50.0

    if target_range:
        lo, hi = target_range
        if lo <= count <= hi:
            score = 35.0 - abs((lo + hi) / 2.0 - count)
        else:
            score = 20.0 - min(abs(count - lo), abs(count - hi)) * 7.0
    elif 3 <= count <= 6:
        score = 25.0 - abs(4.5 - count)
    else:
        score = 10.0 - abs(5 - count) * 4.0

    widths = np.array([c["bbox"][2] for c in chars], dtype=np.float32)
    heights = np.array([c["bbox"][3] for c in chars], dtype=np.float32)
    areas = widths * heights
    med_h = max(1.0, float(np.median(heights)))
    med_w = max(1.0, float(np.median(widths)))

    if H * 0.35 <= med_h <= H * 0.92:
        score += 8.0
    else:
        score -= 8.0
    score -= min(8.0, float(np.std(widths) / med_w) * 3.0)
    score -= min(8.0, float(np.std(heights) / med_h) * 3.0)

    for c, area in zip(chars, areas):
        x, _, w, h = c["bbox"]
        if area < H * W * 0.006:
            score -= 3.0
        if (x <= 1 or x + w >= W - 1) and w < W * 0.07:
            score -= 4.0
        if h < med_h * 0.45:
            score -= 4.0
    return score


def _merge_line_candidates(binary_line, candidates):
    """
    Merge disconnected strokes that belong to the same character.

    Low-quality two-line plates often break a digit such as '5' into separate
    upper/lower components. Merging overlapping or near-overlapping components
    keeps that digit as one OCR crop.
    """
    H, W = binary_line.shape
    filtered = [c for c in candidates if not _is_edge_artifact(c, (H, W))]
    filtered.sort(key=lambda c: c["bbox"][0])

    merged = []
    for cand in filtered:
        x, y, w, h = cand["bbox"]
        if not merged:
            merged.append([x, y, x + w, y + h])
            continue

        lx1, ly1, lx2, ly2 = merged[-1]
        gap = x - lx2
        x_overlap = min(lx2, x + w) - max(lx1, x)
        y_overlap = min(ly2, y + h) - max(ly1, y)
        close_gap = gap <= max(2, int(W * 0.018))
        likely_same_char = (x_overlap > 0 and y_overlap > -H * 0.08) or (
            close_gap and y_overlap > H * 0.12
        )

        if likely_same_char:
            merged[-1] = [
                min(lx1, x),
                min(ly1, y),
                max(lx2, x + w),
                max(ly2, y + h),
            ]
        else:
            merged.append([x, y, x + w, y + h])

    chars = []
    for x1, y1, x2, y2 in merged:
        w = x2 - x1
        h = y2 - y1
        if w < W * 0.018 or h < H * 0.25:
            continue
        char_img = binary_line[y1:y2, x1:x2]
        chars.append({
            "img": char_img,
            "bbox": (x1, y1, w, h),
            "x_center": x1 + w / 2,
            "y_center": y1 + h / 2,
        })
    return chars


def segment_line_components(binary_line, y_offset=0, target_range=None):
    """
    Segment one text row using line-relative geometry.
    """
    H, W = binary_line.shape
    candidate_sets = []

    candidates = component_char_candidates(binary_line)
    candidate_sets.append(_merge_line_candidates(binary_line, candidates))
    candidate_sets.append(
        segment_characters(
            binary_line,
            char_min_width_ratio=0.012,
            char_max_width_ratio=0.50,
            char_min_height_ratio=0.18,
        )
    )
    candidate_sets.append(
        segment_characters_by_projection(
            binary_line,
            min_width_ratio=0.010,
            max_width_ratio=0.34,
            min_height_ratio=0.18,
        )
    )

    best_chars = []
    best_score = -1e9
    for chars in candidate_sets:
        chars = _filter_line_artifacts(chars, (H, W))
        chars.sort(key=lambda c: c["bbox"][0])
        target_max = target_range[1] if target_range else 9
        chars = split_wide_char_candidates(binary_line, chars, target_max=target_max)
        chars = _filter_line_artifacts(chars, (H, W))
        score = _score_line_chars(chars, (H, W), target_range=target_range)
        if score > best_score:
            best_score = score
            best_chars = chars

    adjusted = []
    for c in best_chars:
        x, y, w, h = c["bbox"]
        item = dict(c)
        item["bbox"] = (x, y + y_offset, w, h)
        item["y_center"] = y + y_offset + h / 2
        item["x_center"] = x + w / 2
        adjusted.append(item)
    return adjusted


def trim_outer_artifacts(chars, line_width, min_count=8):
    """
    Drop narrow outer fragments when segmentation returns too many characters.
    """
    if len(chars) <= min_count:
        return chars
    widths = np.array([c["bbox"][2] for c in chars], dtype=np.float32)
    median_w = float(np.median(widths)) if len(widths) else 0.0
    trimmed = list(chars)

    while len(trimmed) > min_count and trimmed:
        x, _, w, _ = trimmed[0]["bbox"]
        if x <= line_width * 0.03 and median_w > 0 and w < median_w * 0.65:
            trimmed.pop(0)
        else:
            break

    while len(trimmed) > min_count and trimmed:
        x, _, w, _ = trimmed[-1]["bbox"]
        if x + w >= line_width * 0.97 and median_w > 0 and w < median_w * 0.65:
            trimmed.pop()
        else:
            break

    return trimmed


def resize_pad_char(char_img, size=32, pad_ratio=0.15):
    """
    Resize ký tự về kích thước (size x size) với padding để giữ aspect ratio.
    
    Args:
        char_img: np.array (ảnh ký tự nhị phân)
        size: kích thước đích (mặc định 32)
        pad_ratio: tỉ lệ padding xung quanh ký tự
        
    Returns:
        np.array: ảnh ký tự size x size
    """
    h, w = char_img.shape[:2]
    target = int(size * (1 - 2 * pad_ratio))
    if h > w:
        new_h = target
        new_w = max(1, int(w * target / h))
    else:
        new_w = target
        new_h = max(1, int(h * target / w))
    resized = cv2.resize(char_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Padding
    canvas = np.zeros((size, size), dtype=np.uint8)
    x_off = (size - new_w) // 2
    y_off = (size - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def _segment_binary_plate(binary, plate_type):
    if plate_type == "2line":
        return segment_two_line_components(binary)

    # One-line plates often include a dark frame connected to the glyphs.
    # A larger vertical inner crop removes that frame; horizontal crop stays
    # small so tight YOLO boxes do not cut off the first/last digit.
    inner = crop_inner_plate(binary, margin_x=0.02, margin_y=0.18)
    all_chars = component_char_candidates(inner)
    all_chars = _filter_line_artifacts(all_chars, inner.shape)
    if len(all_chars) < 7:
        projected_chars = segment_characters_by_projection(inner)
        projected_chars = _filter_line_artifacts(projected_chars, inner.shape)
        if len(projected_chars) > len(all_chars):
            all_chars = projected_chars
    if len(all_chars) < 4:
        all_chars = segment_characters(inner)
    else:
        all_chars.sort(key=lambda c: c["bbox"][0])
        all_chars = split_wide_char_candidates(inner, all_chars, target_max=9)
        all_chars = _filter_line_artifacts(all_chars, inner.shape)
        all_chars = trim_outer_artifacts(all_chars, inner.shape[1], min_count=9)
    return all_chars


def _segmentation_score(binary, chars, plate_type, binary_score):
    count = len(chars)
    fill = float(np.count_nonzero(binary)) / binary.size
    score = binary_score * 0.20

    if 7 <= count <= 9:
        score += 40.0 - abs(8 - count) * 2.0
    elif 5 <= count <= 10:
        score += 18.0 - abs(8 - count) * 3.0
    elif count > 10:
        score += 8.0 - (count - 10) * 3.0
    else:
        score -= (5 - count) * 7.0

    if 0.035 <= fill <= 0.30:
        score += 5.0
    elif fill > 0.42:
        score -= 12.0 + (fill - 0.42) * 25.0

    if chars:
        widths = np.array([c["bbox"][2] for c in chars], dtype=np.float32)
        heights = np.array([c["bbox"][3] for c in chars], dtype=np.float32)
        if len(widths) >= 4:
            med_w = max(1.0, float(np.median(widths)))
            med_h = max(1.0, float(np.median(heights)))
            score -= min(8.0, float(np.std(widths) / med_w) * 4.0)
            score -= min(8.0, float(np.std(heights) / med_h) * 4.0)

    if plate_type == "2line" and chars and all("y_center" in c for c in chars):
        y_min = min(c["y_center"] for c in chars)
        y_max = max(c["y_center"] for c in chars)
        split_y = (y_min + y_max) / 2
        top_count = sum(1 for c in chars if c["y_center"] < split_y)
        bottom_count = count - top_count
        if 2 <= top_count <= 4:
            score += 8.0
        if 4 <= bottom_count <= 6:
            score += 8.0
        if bottom_count >= top_count:
            score += 3.0

    return score


def segment_plate(plate_img, plate_type):
    """
    End-to-end character segmentation cho một plate.
    
    Args:
        plate_img: np.array (ảnh biển số)
        plate_type: "1line" hoặc "2line"
        
    Returns:
        tuple: (char_imgs, all_chars_info, binary_plate)
               - char_imgs: list of np.array (size x size, nhị phân, chuẩn hoá kích thước)
               - all_chars_info: list of dict {"img": ..., "bbox": ...} (gốc chưa resize)
               - binary_plate: ảnh biển số nhị phân hoàn chỉnh
    """
    gray, legacy_binary = preprocess_plate_legacy(plate_img)
    legacy_chars = _segment_binary_plate(legacy_binary, plate_type)

    min_primary_chars = 7 if plate_type == "1line" else 6
    if len(legacy_chars) >= min_primary_chars:
        binary = legacy_binary
        all_chars = legacy_chars
    else:
        gray, candidates = preprocess_plate_candidates(plate_img)
        if not candidates:
            binary = legacy_binary
            all_chars = legacy_chars
        else:
            best = (
                _segmentation_score(
                    legacy_binary,
                    legacy_chars,
                    plate_type,
                    _binary_component_score(legacy_binary),
                ),
                legacy_binary,
                legacy_chars,
            )
            for _, candidate_binary, binary_score in candidates[:10]:
                chars = _segment_binary_plate(candidate_binary, plate_type)
                score = _segmentation_score(candidate_binary, chars, plate_type, binary_score)
                if score > best[0]:
                    best = (score, candidate_binary, chars)
            _, binary, all_chars = best

    # Resize tất cả về 32x32
    char_imgs = [resize_pad_char(c["img"], size=32) for c in all_chars]
    return char_imgs, all_chars, binary
