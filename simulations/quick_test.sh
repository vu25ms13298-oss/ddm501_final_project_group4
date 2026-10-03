#!/usr/bin/env bash
# Quick health and smoke check script for Linux / macOS
set -e

echo "========================================"
echo " LPR STACK END-TO-END HEALTH CHECK"
echo "========================================"

check_service() {
    name="$1"
    url="$2"
    if curl -s -f -m 5 "$url" > /dev/null; then
        echo -e "  \033[32m[OK]\033[0m $name is healthy"
    else
        echo -e "  \033[31m[FAIL]\033[0m $name is not reachable ($url)"
    fi
}

check_service "LPR API" "http://localhost:8000/health"
check_service "MLflow Registry" "http://localhost:5000/api/2.0/mlflow/experiments/search?max_results=1"
check_service "MinIO Storage" "http://localhost:9000/minio/health/live"
check_service "Airflow Webserver" "http://localhost:8080/health"
check_service "Prometheus" "http://localhost:9090/-/healthy"
check_service "Grafana" "http://localhost:3000/api/health"

echo -e "\nSending test prediction to API..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/run_simulation.py" -n 5 -s normal

echo -e "\n\033[32mSmoke tests completed successfully!\033[0m"
