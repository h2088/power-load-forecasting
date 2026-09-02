param(
    [string]$HostAddress = "127.0.0.1",
    [int]$BackendPort = 5000,
    [int]$FrontendPort = 8501,
    [switch]$Visible
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$runtimeDir = Join-Path $root "artifacts\runtime"
$pidFile = Join-Path $runtimeDir "service_pids.json"
$windowStyle = if ($Visible) { "Normal" } else { "Hidden" }

function Test-PortBusy {
    param([int]$Port)
    return $null -ne (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1)
}

function Wait-ForUrl {
    param(
        [string]$Url,
        [int]$MaxAttempts = 20,
        [int]$DelaySeconds = 1
    )

    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing $Url -TimeoutSec 5
            if ($response.StatusCode -eq 200) {
                return $true
            }
        } catch {
        }
        Start-Sleep -Seconds $DelaySeconds
    }

    return $false
}

if (Test-PortBusy -Port $BackendPort) {
    throw "Backend port $BackendPort is already in use. Please stop the existing service first."
}

if (Test-PortBusy -Port $FrontendPort) {
    throw "Frontend port $FrontendPort is already in use. Please stop the existing service first."
}

New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null

$backend = Start-Process `
    -FilePath python `
    -ArgumentList @("-m", "backend.app") `
    -WorkingDirectory $root `
    -WindowStyle $windowStyle `
    -PassThru

if (-not (Wait-ForUrl -Url "http://$HostAddress`:$BackendPort/health")) {
    try {
        Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
    } catch {
    }
    throw "Backend failed to start on http://$HostAddress`:$BackendPort/health"
}

$frontend = Start-Process `
    -FilePath python `
    -ArgumentList @(
        "-m",
        "streamlit",
        "run",
        "frontend/frontend.py",
        "--server.headless",
        "true",
        "--server.address",
        $HostAddress,
        "--server.port",
        "$FrontendPort"
    ) `
    -WorkingDirectory $root `
    -WindowStyle $windowStyle `
    -PassThru

if (-not (Wait-ForUrl -Url "http://$HostAddress`:$FrontendPort")) {
    try {
        Stop-Process -Id $frontend.Id -Force -ErrorAction SilentlyContinue
    } catch {
    }
    try {
        Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
    } catch {
    }
    throw "Frontend failed to start on http://$HostAddress`:$FrontendPort"
}

$runtimeInfo = @{
    backend_pid = $backend.Id
    frontend_pid = $frontend.Id
    backend_health_url = "http://$HostAddress`:$BackendPort/health"
    frontend_url = "http://$HostAddress`:$FrontendPort"
    started_at = (Get-Date).ToString("o")
} | ConvertTo-Json

Set-Content -LiteralPath $pidFile -Value $runtimeInfo -Encoding UTF8

Write-Host ""
Write-Host "Services started successfully."
Write-Host "Backend PID:  $($backend.Id)"
Write-Host "Frontend PID: $($frontend.Id)"
Write-Host "Backend:      http://$HostAddress`:$BackendPort/health"
Write-Host "Frontend:     http://$HostAddress`:$FrontendPort"
Write-Host ""
Write-Host "To stop both services later, run:"
Write-Host "  .\stop.ps1"
