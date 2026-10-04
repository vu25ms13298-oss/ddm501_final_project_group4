#!/usr/bin/env python3
"""CLI utility to execute realistic traffic patterns and drift scenarios against the LPR API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulations.simulator import LPRTrafficSimulator


def parse_args():
    p = argparse.ArgumentParser(
        description="LPR Model Serving Traffic & Drift Simulator"
    )
    p.add_argument(
        "-n",
        "--requests",
        type=int,
        default=60,
        help="Number of prediction requests to send",
    )
    p.add_argument(
        "-s",
        "--scenario",
        type=str,
        default="normal",
        help="Scenario name from config.yaml",
    )
    p.add_argument(
        "--rps",
        type=float,
        default=None,
        help="Target requests per second (default: config value ~0.8)",
    )
    p.add_argument("--concurrency", type=int, default=2, help="Worker threads")
    p.add_argument(
        "--source",
        choices=["dataset", "synthetic", "mixed", "noise_images"],
        default=None,
    )
    p.add_argument(
        "--pattern",
        choices=["steady", "burst", "ramp"],
        default=None,
        help="Run traffic pattern",
    )
    p.add_argument(
        "--bad-requests",
        type=int,
        default=0,
        help="Send N malformed requests to test error metrics",
    )
    p.add_argument("--api-url", type=str, default=None, help="Override API base URL")
    p.add_argument(
        "--config", type=Path, default=None, help="Path to custom config.yaml"
    )
    p.add_argument(
        "--no-respect-rate-limit",
        action="store_true",
        help="Do not sleep on 429 Retry-After",
    )
    p.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List available scenarios and exit",
    )
    return p.parse_args()


def main():
    args = parse_args()
    sim = LPRTrafficSimulator(config_path=args.config)

    if args.api_url:
        sim.base_url = args.api_url.rstrip("/")
        sim.predict_url = f"{sim.base_url}/predict"
        sim.health_url = f"{sim.base_url}/health"

    if args.list_scenarios:
        print("\nAvailable simulation scenarios in config.yaml:")
        for name, info in sim.scenarios.items():
            print(f"  • {name:<16}: {info.get('description', '')}")
        return 0

    if not sim.check_api_health():
        print("Aborting: API is not healthy or unreachable.")
        return 1

    if args.bad_requests > 0:
        sim.send_bad_requests(args.bad_requests)

    if args.pattern:
        sim.run_traffic_pattern(pattern=args.pattern, scenario=args.scenario)
    else:
        rps = args.rps or sim.config.get("defaults", {}).get("rps", 0.8)
        source = args.source or sim.config.get("defaults", {}).get("source", "mixed")
        sim.run_simulation(
            n_requests=args.requests,
            scenario=args.scenario,
            rps=rps,
            concurrency=args.concurrency,
            source=source,
            respect_rate_limit=not args.no_respect_rate_limit,
        )

    sim.print_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
