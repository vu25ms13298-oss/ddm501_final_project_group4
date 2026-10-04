"""LPR Traffic Simulator.

Executes concurrent and rate-controlled HTTP requests against the LPR API,
recording telemetry, evaluating drift responses, and capturing Prometheus snapshots.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import requests
import yaml
from colorama import Fore, Style, init

from simulations.drift_transforms import apply_pipeline
from simulations.image_source import ImageSource, Sample

# Windows console UTF-8 setup
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

init(autoreset=True)
log = logging.getLogger("lpr_simulator")


class LPRTrafficSimulator:
    """Manages traffic generation, scenarios, and drift induction."""

    def __init__(self, config_path: str | Path | None = None):
        if config_path is None:
            config_path = Path(__file__).resolve().parent / "config.yaml"
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.api_cfg = self.config.get("api", {})
        self.base_url = self.api_cfg.get("base_url", "http://localhost:8000")
        self.predict_url = f"{self.base_url}{self.api_cfg.get('predict_path', '/predict')}"
        self.health_url = f"{self.base_url}{self.api_cfg.get('health_path', '/health')}"
        self.prom_url = self.config.get("prometheus", {}).get("url", "http://localhost:9090")

        self.image_source = ImageSource(self.config)
        self.scenarios = self.config.get("scenarios", {})
        self.lock = threading.Lock()
        self.reset_stats()

    def reset_stats(self):
        with self.lock:
            self.stats = {
                "total": 0,
                "success_2xx": 0,
                "error_4xx": 0,
                "rate_limited_429": 0,
                "error_5xx": 0,
                "exceptions": 0,
                "latencies_ms": [],
                "format_valid": 0,
                "exact_matches": 0,
                "labeled_count": 0,
            }

    def check_api_health(self) -> bool:
        """Verifies API readiness and prints loaded model details."""
        try:
            r = requests.get(self.health_url, timeout=5)
            if r.status_code != 200:
                print(f"{Fore.RED}✗ API health check failed: HTTP {r.status_code}{Style.RESET_ALL}")
                return False
            data = r.json()
            loaded = data.get("model_loaded", False)
            status = data.get("status")

            info_resp = requests.get(f"{self.base_url}/model/info", timeout=5)
            info = info_resp.json() if info_resp.status_code == 200 else {}

            source = info.get("model_source", "unknown")
            version = info.get("model_version") or "baked-in"
            yolo = "YOLOv8" if info.get("yolo_loaded") else "Contour fallback"

            print(f"{Fore.GREEN}✓ API is {status} | Model: {source} (v{version}) | Detector: {yolo}{Style.RESET_ALL}")
            return loaded
        except Exception as exc:
            print(f"{Fore.RED}✗ Cannot connect to API at {self.health_url}: {exc}{Style.RESET_ALL}")
            return False

    def send_prediction(
        self,
        sample: Sample,
        transforms: list[dict[str, Any]] | None = None,
        respect_rate_limit: bool = True,
    ) -> dict[str, Any]:
        """Applies transforms, sends multipart /predict request and updates metrics."""
        img = sample.image
        if transforms:
            img = apply_pipeline(img, transforms)

        success_encode, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if not success_encode:
            return {"status": "encode_error"}

        files = {"file": (sample.name, buf.tobytes(), "image/jpeg")}
        data = {"assume_plate_crop": "true" if sample.is_crop else "false"}

        t0 = time.time()
        status_code = 0
        resp_json = {}
        try:
            resp = requests.post(self.predict_url, files=files, data=data, timeout=30)
            latency_ms = (time.time() - t0) * 1000
            status_code = resp.status_code

            if status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 5))
                if respect_rate_limit:
                    time.sleep(retry_after)
            elif status_code == 200:
                resp_json = resp.json()

        except Exception as exc:
            latency_ms = (time.time() - t0) * 1000
            with self.lock:
                self.stats["total"] += 1
                self.stats["exceptions"] += 1
            return {"status": "exception", "error": str(exc), "latency_ms": latency_ms}

        with self.lock:
            self.stats["total"] += 1
            self.stats["latencies_ms"].append(latency_ms)

            if 200 <= status_code < 300:
                self.stats["success_2xx"] += 1
                if resp_json.get("success"):
                    self.stats["format_valid"] += 1
                if sample.label:
                    self.stats["labeled_count"] += 1
                    pred_clean = (resp_json.get("plate_text") or "").replace("-", "").replace(".", "").upper()
                    target_clean = sample.label.replace("-", "").replace(".", "").upper()
                    if pred_clean == target_clean:
                        self.stats["exact_matches"] += 1
            elif status_code == 429:
                self.stats["rate_limited_429"] += 1
            elif 400 <= status_code < 500:
                self.stats["error_4xx"] += 1
            elif status_code >= 500:
                self.stats["error_5xx"] += 1

        return {
            "status_code": status_code,
            "latency_ms": latency_ms,
            "response": resp_json,
        }

    def run_simulation(
        self,
        n_requests: int = 60,
        scenario: str = "normal",
        rps: float = 0.8,
        concurrency: int = 2,
        source: str = "mixed",
        respect_rate_limit: bool = True,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """Runs controlled simulation with target RPS and concurrency."""
        from tqdm import tqdm

        sc_info = self.scenarios.get(scenario, {})
        transforms = sc_info.get("transforms", [])
        src = sc_info.get("source", source)

        desc = f"Simulating '{scenario}' ({n_requests} reqs @ {rps} rps)"
        interval = 1.0 / max(0.1, float(rps))

        pbar = tqdm(total=n_requests, desc=desc, disable=not show_progress)
        futures = []

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for _ in range(n_requests):
                sample = self.image_source.sample(src)
                f = executor.submit(self.send_prediction, sample, transforms, respect_rate_limit)
                futures.append(f)
                time.sleep(interval)
                pbar.update(1)

            for f in as_completed(futures):
                pass
        pbar.close()
        return dict(self.stats)

    def run_traffic_pattern(self, pattern: str = "burst", scenario: str = "normal", duration_s: int = 15):
        """Runs traffic pattern (burst, ramp, or steady)."""
        print(f"\n{Fore.CYAN}▶ Running traffic pattern: {pattern.upper()} (duration {duration_s}s){Style.RESET_ALL}")
        if pattern == "burst":
            # Fire rapid burst without pacing to trigger rate limit (429)
            self.run_simulation(n_requests=35, scenario=scenario, rps=8.0, concurrency=4, respect_rate_limit=False)
        elif pattern == "ramp":
            for rps in (0.5, 1.0, 2.0, 4.0):
                print(f"  Ramping to {rps} RPS...")
                self.run_simulation(n_requests=10, scenario=scenario, rps=rps, concurrency=2, show_progress=False)
        else:
            self.run_simulation(n_requests=int(duration_s * 0.8), scenario=scenario, rps=0.8, concurrency=2)

    def send_bad_requests(self, n: int = 10):
        """Sends deliberately malformed requests to populate error metrics."""
        print(f"\n{Fore.YELLOW}▶ Sending {n} invalid payloads (empty, non-image, oversized)...{Style.RESET_ALL}")
        for i in range(n):
            mode = i % 3
            if mode == 0:
                # Empty file -> 400
                files = {"file": ("empty.jpg", b"", "image/jpeg")}
            elif mode == 1:
                # Corrupted non-image bytes -> 400
                files = {"file": ("corrupt.jpg", b"NOT_AN_IMAGE_PAYLOAD_12345678", "image/jpeg")}
            else:
                # Oversized > 10MB -> 413
                files = {"file": ("huge.jpg", b"0" * (11 * 1024 * 1024), "image/jpeg")}
            try:
                requests.post(self.predict_url, files=files, timeout=5)
            except Exception:
                pass
        print(f"{Fore.GREEN}✓ Bad requests sent.{Style.RESET_ALL}")

    def prometheus_snapshot(self) -> dict[str, Any]:
        """Queries Prometheus for instantaneous drift and health values."""
        queries = {
            "success_rate_5m": 'sum(rate(plate_recognition_result_total{result="success"}[5m])) / sum(rate(plate_recognition_result_total[5m]))',
            "brightness_mean_5m": "rate(input_image_brightness_sum[5m]) / rate(input_image_brightness_count[5m])",
            "width_mean_5m": "rate(input_image_width_pixels_sum[5m]) / rate(input_image_width_pixels_count[5m])",
            "active_alerts": 'count(ALERTS{alertstate="firing"})',
        }
        res = {}
        for k, q in queries.items():
            try:
                r = requests.get(f"{self.prom_url}/api/v1/query", params={"query": q}, timeout=3)
                if r.status_code == 200:
                    data = r.json().get("data", {}).get("result", [])
                    if data:
                        res[k] = round(float(data[0]["value"][1]), 3)
            except Exception:
                pass
        return res

    def print_summary(self):
        """Prints formatted summary statistics."""
        s = self.stats
        total = s["total"]
        if total == 0:
            print("No requests executed.")
            return

        lats = s["latencies_ms"]
        p50 = round(sorted(lats)[len(lats) // 2], 1) if lats else 0
        p95 = round(sorted(lats)[int(len(lats) * 0.95)], 1) if lats else 0

        acc_str = "N/A"
        if s["labeled_count"] > 0:
            acc_pct = (s["exact_matches"] / s["labeled_count"]) * 100
            acc_str = f"{acc_pct:.1f}% ({s['exact_matches']}/{s['labeled_count']})"

        print(f"\n{Fore.CYAN}{'='*60}")
        print("SIMULATION RUN SUMMARY")
        print(f"{'='*60}{Style.RESET_ALL}")
        print(f"Total Requests     : {total}")
        print(f"Success (200 OK)   : {Fore.GREEN}{s['success_2xx']}{Style.RESET_ALL}")
        print(f"Format Valid       : {Fore.GREEN}{s['format_valid']}{Style.RESET_ALL}")
        print(f"Rate Limited (429) : {Fore.YELLOW if s['rate_limited_429'] else Fore.WHITE}{s['rate_limited_429']}{Style.RESET_ALL}")
        print(f"Client Errors (4xx): {Fore.YELLOW if s['error_4xx'] else Fore.WHITE}{s['error_4xx']}{Style.RESET_ALL}")
        print(f"Server Errors (5xx): {Fore.RED if s['error_5xx'] else Fore.WHITE}{s['error_5xx']}{Style.RESET_ALL}")
        print(f"Latency P50 / P95  : {p50} ms / {p95} ms")
        print(f"Plate Exact Match  : {acc_str}")

        snap = self.prometheus_snapshot()
        if snap:
            print(f"\n{Fore.MAGENTA}Prometheus Instant Telemetry:{Style.RESET_ALL}")
            for k, v in snap.items():
                print(f"  • {k:<20}: {v}")
        print(f"{Fore.CYAN}{'='*60}\n{Style.RESET_ALL}")
