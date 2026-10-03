# Quick health and smoke check script for Windows PowerShell
$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " LPR STACK END-TO-END HEALTH CHECK" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$services = @(
    @{ Name = "LPR API"; Url = "http://localhost:8000/health" },
    @{ Name = "MLflow Registry"; Url = "http://localhost:5000/api/2.0/mlflow/experiments/search?max_results=1" },
    @{ Name = "MinIO Storage"; Url = "http://localhost:9000/minio/health/live" },
    @{ Name = "Airflow Webserver"; Url = "http://localhost:8080/health" },
    @{ Name = "Prometheus Metrics"; Url = "http://localhost:9090/-/healthy" },
    @{ Name = "Grafana Dashboards"; Url = "http://localhost:3000/api/health" }
)

foreach ($s in $services) {
    try {
        $resp = Invoke-RestMethod -Uri $s.Url -Method Get -TimeoutSec 5
        Write-Host "  [OK] $($s.Name) is healthy" -ForegroundColor Green
    } catch {
        Write-Host "  [FAIL] $($s.Name) is not reachable ($($_.Exception.Message))" -ForegroundColor Red
    }
}

Write-Host "`nSending test prediction to API..." -ForegroundColor Cyan
try {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    python "$scriptDir\run_simulation.py" -n 5 -s normal
    Write-Host "`nAll smoke tests completed!" -ForegroundColor Green
} catch {
    Write-Host "`nSimulation run encountered an error: $($_.Exception.Message)" -ForegroundColor Red
}
