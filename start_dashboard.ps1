$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$dashboardRoot = Join-Path $projectRoot "dashboard"
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$portableNode = Join-Path $projectRoot "tmp\node22\node-v22.13.0-win-x64"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python virtual environment not found: $python"
}

if (Test-Path -LiteralPath (Join-Path $portableNode "node.exe")) {
    $env:PATH = "$portableNode;$env:PATH"
}

$nodeVersion = (& node --version 2>$null)
if (-not $nodeVersion -or [int](($nodeVersion -replace '^v','').Split('.')[0]) -lt 22) {
    throw "Node.js 22.13+ is required."
}

$apiProcess = $null
try {
    $null = Invoke-RestMethod -Uri "http://127.0.0.1:8765/api/health" -TimeoutSec 1
    Write-Host "Dashboard API is already running"
}
catch {
    $apiProcess = Start-Process -FilePath $python `
        -ArgumentList (Join-Path $projectRoot "dashboard_server.py") `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden `
        -PassThru
    Write-Host "Dashboard API started (PID $($apiProcess.Id))"
}

Write-Host "Agent Observatory: http://localhost:3000"
Write-Host "Press Ctrl+C to stop."

try {
    Set-Location -LiteralPath $dashboardRoot
    & npm run dev
}
finally {
    if ($null -ne $apiProcess -and -not $apiProcess.HasExited) {
        Stop-Process -Id $apiProcess.Id
    }
}
