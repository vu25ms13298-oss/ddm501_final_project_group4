"""Unit tests for plate detection (YOLO wrapper with a fake model + contour fallback)."""

import cv2
import numpy as np

from src.detection import detect_plate_contour_fallback, detect_plate_yolo


class _Tensor:
    def __init__(self, values):
        self._values = np.asarray(values)

    def __getitem__(self, idx):
        return _Tensor(self._values[idx])

    def cpu(self):
        return self

    def numpy(self):
        return self._values

    def __float__(self):
        return float(self._values)


class _Box:
    def __init__(self, xyxy, conf):
        self.xyxy = _Tensor([xyxy])
        self.conf = _Tensor([conf])


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeYOLO:
    def __init__(self, results):
        self.results = results
        self.kwargs = None

    def predict(self, image, **kwargs):
        self.kwargs = kwargs
        return self.results


class TestYOLOWrapper:
    def test_returns_boxes_with_confidence(self):
        model = _FakeYOLO([_Result([_Box([10.7, 20.2, 110.9, 60.1], 0.87)])])
        boxes = detect_plate_yolo(np.zeros((100, 200, 3), np.uint8), model, 0.3)
        assert boxes == [(10, 20, 110, 60, 0.87)]
        assert model.kwargs["conf"] == 0.3

    def test_skips_results_without_boxes(self):
        model = _FakeYOLO([_Result(None), _Result([])])
        assert detect_plate_yolo(np.zeros((10, 10, 3), np.uint8), model) == []


class TestContourFallback:
    def test_finds_plate_like_rectangle(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (150, 200), (450, 290), (255, 255, 255), thickness=3)
        candidates = detect_plate_contour_fallback(img)
        assert candidates
        x1, y1, x2, y2, ar = candidates[0]
        assert 1.2 <= ar <= 5.5
        assert x1 < x2 and y1 < y2

    def test_blank_image_has_no_candidates(self):
        assert detect_plate_contour_fallback(np.zeros((200, 200, 3), np.uint8)) == []

    def test_ignores_square_shapes(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (300, 300), (255, 255, 255), thickness=3)
        assert detect_plate_contour_fallback(img) == []
