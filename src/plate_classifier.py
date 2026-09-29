# ==========================================
# Member 3: Section 4 — Plate Type Classification (1-line vs 2-line)
# ==========================================
# This module classifies license plates based on aspect ratio (AR) of width / height.

def classify_plate_type(plate_img, threshold=2.5):
    """
    Phân loại loại biển số dựa trên aspect ratio.

    Args:
        plate_img: ảnh biển số đã được crop (numpy array)
        threshold: ngưỡng AR phân biệt 1-line/2-line (ví dụ 2.5)

    Returns:
        tuple: (plate_type, aspect_ratio) 
               - plate_type: "1line" (biển dài) hoặc "2line" (biển vuông/2 dòng)
               - aspect_ratio: tỉ lệ rộng/cao thực tế
    """
    h, w = plate_img.shape[:2]
    aspect_ratio = w / h
    plate_type = "1line" if aspect_ratio > threshold else "2line"
    return plate_type, aspect_ratio
