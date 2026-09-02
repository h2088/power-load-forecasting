$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pidFile = Join-Path $root "artifacts\runtime\service_pids.json"

if (-not (Test-Path -LiteralPath $pidFile)) {
    Write-Host "No runtime pid file found. Nothing to stop."
    exit 0
}

$runtimeInfo = Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json
$stoppedAny = $false

foreach ($pidValue in @($runtimeInfo.backend_pid, $runtimeInfo.frontend_pid)) {
    if (-not $pidValue) {
        continue
    }

    try {
        $process = Get-Process -Id $pidValue -ErrorAction Stop
        Stop-Process -Id $process.Id -Force
        Write-Host "Stopped process $($process.Id) ($($process.ProcessName))."
        $stoppedAny = $true
    } catch {
        Write-Host "Process $pidValue is not running."
    }
}

Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue

if (-not $stoppedAny) {
    Write-Host "No running frontend/backend processes were found."
}
