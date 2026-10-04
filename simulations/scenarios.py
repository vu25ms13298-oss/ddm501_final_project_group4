#!/usr/bin/env python3
"""Pre-configured end-to-end simulation scenarios for demonstrations and monitoring tests.

Scenarios test:
1. Normal daytime traffic (baseline establishment)
2. Gradual environmental drift (dusk -> nightfall brightness drop)
3. Sudden hardware shift (low-resolution camera swap)
4. Weather fluctuation (rain & mud occlusion mix)
5. Prediction drift (non-plate images causing format degradation)
6. Rate limiting & traffic spike (triggers HTTP 429)
7. Client error telemetry (empty & oversized payloads)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from colorama import Fore, Style, init

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulations.simulator import LPRTrafficSimulator

init(autoreset=True)


def banner(title: str, subtitle: str = ""):
    print(f"\n{Fore.CYAN}{'='*64}")
    print(f" {title.upper()}")
    if subtitle:
        print(f" {Fore.WHITE}{subtitle}")
    print(f"{Fore.CYAN}{'='*64}{Style.RESET_ALL}\n")


def scenario_1_normal():
    banner("Scenario 1: Normal Daytime Traffic", "Simulates steady, high-quality camera captures (100 reqs)")
    sim = LPRTrafficSimulator()
    if not sim.check_api_health():
        return
    sim.run_simulation(n_requests=100, scenario="normal", rps=0.8, concurrency=2)
    sim.print_summary()
    print(f"{Fore.GREEN}✓ Observe Grafana: RPS ~ 0.8, P95 < 500ms, Success Rate ~ 100%, No Alerts firing.{Style.RESET_ALL}")


def scenario_2_nightfall():
    banner("Scenario 2: Nightfall (Gradual Data Drift)", "Daylight transitions into twilight and severe night darkness")
    sim = LPRTrafficSimulator()
    if not sim.check_api_health():
        return

    print(f"\n{Fore.YELLOW}Phase 1: Standard Daylight (40 requests)...{Style.RESET_ALL}")
    sim.run_simulation(n_requests=40, scenario="normal", rps=0.8)

    print(f"\n{Fore.YELLOW}Phase 2: Evening Twilight / Dusk (40 requests)...{Style.RESET_ALL}")
    sim.run_simulation(n_requests=40, scenario="dusk", rps=0.8)

    print(f"\n{Fore.YELLOW}Phase 3: Deep Night + Sensor Noise (50 requests)...{Style.RESET_ALL}")
    sim.run_simulation(n_requests=50, scenario="night", rps=0.8)

    sim.print_summary()
    print(f"{Fore.GREEN}✓ Observe Grafana & Prometheus:{Style.RESET_ALL}")
    print("  • Panel 'Input Brightness' drops significantly.")
    print("  • Alert 'InputBrightnessDriftFast' transitions to PENDING -> FIRING.")


def scenario_3_camera_swap():
    banner("Scenario 3: Camera Hardware Swap (Resolution Shift)", "Sudden transition from HD camera to low-res compressed feed")
    sim = LPRTrafficSimulator()
    if not sim.check_api_health():
        return

    print(f"\n{Fore.YELLOW}Phase 1: High-res camera feed (40 requests)...{Style.RESET_ALL}")
    sim.run_simulation(n_requests=40, scenario="normal", rps=0.8)

    print(f"\n{Fore.YELLOW}Phase 2: Swapping to low-res sensor (60 requests)...{Style.RESET_ALL}")
    sim.run_simulation(n_requests=60, scenario="low_res_camera", rps=0.8)

    sim.print_summary()
    print(f"{Fore.GREEN}✓ Observe Grafana & Prometheus:{Style.RESET_ALL}")
    print("  • Panel 'Input Width (5m vs 1h)' plummets.")
    print("  • Alert 'InputResolutionDriftFast' triggers.")


def scenario_4_weather_mix():
    banner("Scenario 4: Adverse Weather Fluctuation", "Alternating rain, lens splatter and occlusion")
    sim = LPRTrafficSimulator()
    if not sim.check_api_health():
        return

    phases = [("normal", 25), ("rain", 35), ("normal", 20), ("occluded", 35), ("normal", 20)]
    for sc, count in phases:
        print(f"\n{Fore.YELLOW}Phase: {sc.upper()} ({count} requests)...{Style.RESET_ALL}")
        sim.run_simulation(n_requests=count, scenario=sc, rps=0.8)

    sim.print_summary()
    print(f"{Fore.GREEN}✓ Observe Grafana: 'Characters Detected per Plate' and format scores dip during rain/occlusion.{Style.RESET_ALL}")


def scenario_5_prediction_drift():
    banner("Scenario 5: Prediction / Concept Drift", "Camera aimed at scenery with no vehicles, degrading plate format scores")
    sim = LPRTrafficSimulator()
    if not sim.check_api_health():
        return

    print(f"\n{Fore.YELLOW}Phase 1: Valid plates (40 requests)...{Style.RESET_ALL}")
    sim.run_simulation(n_requests=40, scenario="normal", rps=0.8)

    print(f"\n{Fore.YELLOW}Phase 2: Non-plate background noise (60 requests)...{Style.RESET_ALL}")
    sim.run_simulation(n_requests=60, scenario="non_plate", rps=0.8)

    sim.print_summary()
    print(f"{Fore.GREEN}✓ Observe Grafana & Prometheus:{Style.RESET_ALL}")
    print("  • Panel 'Recognition Success Rate (5m)' drops below 50%.")
    print("  • Alert 'LowRecognitionSuccessRateFast' triggers.")


def scenario_6_rate_limit_spike():
    banner("Scenario 6: Traffic Spike & Rate Limiting", "Fires rapid requests exceeding 60 req/min to trigger HTTP 429")
    sim = LPRTrafficSimulator()
    if not sim.check_api_health():
        return

    print(f"{Fore.YELLOW}Blasting 70 requests across 4 workers with no backoff...{Style.RESET_ALL}")
    sim.run_simulation(n_requests=70, scenario="normal", rps=10.0, concurrency=4, respect_rate_limit=False)

    sim.print_summary()
    print(f"{Fore.GREEN}✓ Observe Grafana & Prometheus:{Style.RESET_ALL}")
    print("  • Panel '/predict HTTP Status Codes' shows red HTTP 429 line.")
    print("  • Panel 'Rate-limited Requests' records rejected count.")
    print("  • Alert 'RateLimitBurst' fires.")


def scenario_7_bad_inputs():
    banner("Scenario 7: Client Errors Telemetry", "Sends invalid, empty and oversized payloads")
    sim = LPRTrafficSimulator()
    if not sim.check_api_health():
        return

    sim.send_bad_requests(15)
    print(f"{Fore.YELLOW}Following up with 20 normal requests...{Style.RESET_ALL}")
    sim.run_simulation(n_requests=20, scenario="normal", rps=0.8)

    sim.print_summary()
    print(f"{Fore.GREEN}✓ Observe Grafana: Panel 'Prediction Errors' logs empty_file, invalid_image, payload_too_large.{Style.RESET_ALL}")


SCENARIOS = {
    1: ("Normal Day Traffic", scenario_1_normal),
    2: ("Nightfall (Gradual Drift)", scenario_2_nightfall),
    3: ("Camera Swap (Sudden Shift)", scenario_3_camera_swap),
    4: ("Adverse Weather Mix", scenario_4_weather_mix),
    5: ("Prediction Drift (Non-plate)", scenario_5_prediction_drift),
    6: ("Traffic Spike / Rate Limit (429)", scenario_6_rate_limit_spike),
    7: ("Bad Input Errors (400/413)", scenario_7_bad_inputs),
}


def main():
    p = argparse.ArgumentParser(description="Run ready-made LPR simulation scenarios")
    p.add_argument("scenario", type=int, nargs="?", choices=list(SCENARIOS.keys()), help="Scenario number 1-7 (omit to run all)")
    args = p.parse_args()

    if args.scenario:
        name, fn = SCENARIOS[args.scenario]
        print(f"\n{Fore.CYAN}Starting Scenario {args.scenario}: {name}...{Style.RESET_ALL}")
        fn()
    else:
        print(f"\n{Fore.MAGENTA}=== RUNNING ALL 7 LPR SCENARIOS SEQUENTIALLY ==={Style.RESET_ALL}")
        for num, (name, fn) in SCENARIOS.items():
            print(f"\n{Fore.CYAN}[{num}/7] {name}{Style.RESET_ALL}")
            fn()
            if num < len(SCENARIOS):
                print(f"{Fore.YELLOW}Pausing 5 seconds before next scenario...{Style.RESET_ALL}")
                time.sleep(5)
        print(f"\n{Fore.GREEN}=== ALL SCENARIOS COMPLETED ==={Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
