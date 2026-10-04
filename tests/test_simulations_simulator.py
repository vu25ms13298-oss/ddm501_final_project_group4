from unittest.mock import MagicMock, patch
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from simulations.image_source import ImageSource, Sample, make_non_plate_images
from simulations.simulator import LPRTrafficSimulator


@pytest.fixture
def mock_config(tmp_path):
    cfg_file = tmp_path / "test_config.yaml"
    cfg_content = """
api:
  base_url: "http://testserver"
  predict_path: "/predict"
  health_path: "/health"
prometheus:
  url: "http://testserver:9090"
images:
  dataset_dirs: []
  synthetic_pool_size: 5
  synthetic_seed: 42
  scene_fraction: 0.0
defaults:
  rps: 1.0
  concurrency: 1
scenarios:
  normal:
    transforms: []
  dusk:
    transforms:
      - name: brightness
        factor: 0.6
"""
    cfg_file.write_text(cfg_content)
    return cfg_file


def test_image_source_noise():
    samples = make_non_plate_images(5)
    assert len(samples) == 5
    assert samples[0].image.shape == (480, 640, 3)


def test_image_source_sampling(mock_config):
    src = ImageSource({"images": {"dataset_dirs": [], "synthetic_pool_size": 2}})
    s = src.sample("noise_images")
    assert isinstance(s, Sample)
    assert s.image is not None


def test_simulator_stats_reset(mock_config):
    sim = LPRTrafficSimulator(config_path=mock_config)
    assert sim.stats["total"] == 0
    sim.stats["total"] = 5
    sim.reset_stats()
    assert sim.stats["total"] == 0


def test_simulator_send_prediction_mock(mock_config):
    sim = LPRTrafficSimulator(config_path=mock_config)
    sample = Sample(
        name="test.png", image=np.zeros((64, 64, 3), dtype=np.uint8), is_crop=True
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "success": True,
        "plate_text": "51F12345",
        "latency_ms": 50.0,
    }

    with patch("requests.post", return_value=mock_resp):
        res = sim.send_prediction(sample)
        assert res["status_code"] == 200
        assert sim.stats["total"] == 1
        assert sim.stats["success_2xx"] == 1
        assert sim.stats["format_valid"] == 1


def test_simulator_send_prediction_429_retry(mock_config):
    sim = LPRTrafficSimulator(config_path=mock_config)
    sample = Sample(
        name="test.png", image=np.zeros((64, 64, 3), dtype=np.uint8), is_crop=True
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.headers = {"Retry-After": "0"}

    with (
        patch("requests.post", return_value=mock_resp),
        patch("time.sleep") as mock_sleep,
    ):
        res = sim.send_prediction(sample, respect_rate_limit=True)
        assert res["status_code"] == 429
        assert sim.stats["rate_limited_429"] == 1
        mock_sleep.assert_called_once_with(0)
