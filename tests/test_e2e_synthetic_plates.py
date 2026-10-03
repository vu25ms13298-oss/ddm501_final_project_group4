"""End-to-end test: rendered Vietnamese plates -> full LPR pipeline -> plate text.

Uses scripts/generate_synthetic_plates.py so every stage (preprocessing,
detection fallback, segmentation, HOG features, SVM, format correction) runs on
labelled input. Thresholds are deliberately below the measured benchmark scores
so the test catches regressions without being flaky.
"""

import csv
import random

import cv2
import pytest

from scripts.evaluate_lpr_end_to_end import levenshtein, normalize_text
from scripts.generate_synthetic_plates import generate, random_plate
from src.pipeline import LPRPipeline

MIN_CHAR_ACCURACY = 0.80
MIN_EXACT_ACCURACY = 0.40


@pytest.fixture(scope="module")
def benchmark(tmp_path_factory):
    try:
        manifest = generate(
            tmp_path_factory.mktemp("plates"), n=12, seed=7, scene_fraction=0.25
        )
    except RuntimeError as exc:  # no TrueType font available
        pytest.skip(str(exc))
    with manifest.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def lpr(models_dir):
    return LPRPipeline(models_dir=models_dir)


def test_random_plate_formats():
    rng = random.Random(0)
    label, lines = random_plate(rng, two_line=False)
    assert len(label) == 8 and len(lines) == 1
    label, lines = random_plate(rng, two_line=True)
    assert len(label) == 9 and len(lines) == 2


def test_manifest_covers_crops_and_scenes(benchmark):
    crops = {row["plate_crop"] for row in benchmark}
    assert crops == {"True", "False"}


def test_pipeline_reads_synthetic_plates(benchmark, lpr):
    exact, edits, total = 0, 0, 0
    for row in benchmark:
        image = cv2.cvtColor(cv2.imread(row["image"]), cv2.COLOR_BGR2RGB)
        result = lpr.recognize(
            image, assume_plate_crop=row["plate_crop"] == "True", verbose=False
        )
        pred = normalize_text(result.get("plate_string", ""))
        exact += int(pred == row["label"])
        edits += levenshtein(row["label"], pred)
        total += max(len(row["label"]), len(pred), 1)

    char_acc = 1 - edits / total
    exact_acc = exact / len(benchmark)
    assert char_acc >= MIN_CHAR_ACCURACY, f"char accuracy {char_acc:.2%}"
    assert exact_acc >= MIN_EXACT_ACCURACY, f"exact accuracy {exact_acc:.2%}"


def test_verbose_recognize_runs(benchmark, lpr, capsys):
    row = benchmark[0]
    image = cv2.cvtColor(cv2.imread(row["image"]), cv2.COLOR_BGR2RGB)
    result = lpr.recognize(
        image, assume_plate_crop=row["plate_crop"] == "True", verbose=True
    )
    assert isinstance(result, dict)
