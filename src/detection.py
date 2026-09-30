# ==========================================
# Member 3: Section 3 — License Plate Detection (YOLOv8 & Fallback)
# ==========================================
# This module implements license plate detection using YOLOv8,
# with a contour-based fallback algorithm for demo/fallback purposes.

import torch
from ultralytics import YOLO
import cv2


# ============= YOLOv8 Training Template =============
# Chạy khi có dataset từ Member 1 (data.yaml)
def train_yolo(data_yaml_path, epochs=50, device=None):
    """
    Train YOLOv8 on Vietnam license plate dataset.

    Args:
        data_yaml_path: path to data.yaml
        epochs: number of epochs to train
        device: device (e.g. 0, "cpu", etc.)

    Returns:
        str: path to best.pt weights
    """
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    # Load pre-trained YOLOv8n (nano model)
    model = YOLO("yolov8n.pt")

    # Train
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=16,
        name="license_plate_v1",
        patience=10,
        device=device,
    )

    print("✅ Train xong! Best weights tại:", results.save_dir / "weights/best.pt")
    return str(results.save_dir / "weights/best.pt")


# ============= Inference Functions =============


def detect_plate_yolo(image, model, conf_threshold=0.25):
    """
    Detect biển số trong ảnh bằng YOLO.

    Args:
        image: np.array (RGB scene image)
        model: loaded YOLO model
        conf_threshold: confidence threshold for detection

    Returns:
        list of (x1, y1, x2, y2, confidence) — các bbox phát hiện được
    """
    results = model.predict(image, conf=conf_threshold, verbose=False)
    boxes = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            boxes.append((x1, y1, x2, y2, conf))
    return boxes


def detect_plate_contour_fallback(image):
    """
    Fallback method: dùng contour detection khi không có YOLO model train riêng.
    Tìm các vùng hình chữ nhật có aspect ratio phù hợp với biển số Việt Nam.

    Args:
        image: np.array (RGB scene image)

    Returns:
        list of (x1, y1, x2, y2, aspect_ratio) — các candidates
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blurred, 30, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    candidates = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ar = w / float(h)
            area = w * h
            # Biển số Việt Nam: AR ~ 1.2–5.5, diện tích đủ lớn
            if 1.2 <= ar <= 5.5 and area > 5000:
                candidates.append((x, y, x + w, y + h, ar))

    return candidates
