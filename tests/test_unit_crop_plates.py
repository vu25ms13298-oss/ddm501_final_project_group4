"""Unit tests for cropping ground-truth plates from YOLO polygon labels."""

from pathlib import Path

import numpy as np

from scripts.crop_plates_from_labels import (
    label_path_for,
    largest_polygon,
    order_corners,
    warp_plate,
)


def test_label_path_maps_images_to_labels():
    path = Path("dataset/images/val/Dieu_0003.png")
    assert label_path_for(path) == Path("dataset/labels/val/Dieu_0003.txt")


def test_order_corners_is_clockwise_from_top_left():
    shuffled = np.array([[90, 50], [10, 10], [10, 50], [90, 10]], dtype=np.float32)
    np.testing.assert_array_equal(
        order_corners(shuffled), [[10, 10], [90, 10], [90, 50], [10, 50]]
    )


def test_largest_polygon_picks_biggest_plate(tmp_path):
    label = tmp_path / "x.txt"
    label.write_text(
        "1 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n"  # small
        "1 0.4 0.4 0.8 0.4 0.8 0.7 0.4 0.7\n"  # large
        "bad line\n",
        encoding="utf-8",
    )
    poly = largest_polygon(label, width=100, height=100)
    np.testing.assert_allclose(poly.min(axis=0), [40, 40])


def test_warp_plate_returns_upright_crop_with_margin():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[20:60, 40:160] = 255
    poly = np.array([[40, 20], [160, 20], [160, 60], [40, 60]], dtype=np.float32)
    crop = warp_plate(image, poly, margin=0.05)
    assert crop.shape[1] > crop.shape[0]  # stays landscape
    assert crop.shape[1] == int(120 * 1.1)
    assert crop[crop.shape[0] // 2, crop.shape[1] // 2].mean() == 255
