Param(
    [string]$PythonExe = "C:/Users/39583/AppData/Local/Programs/Python/Python38/python.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

$probeJson = & $PythonExe -c "import json,sys,torch;print(json.dumps({'exe':sys.executable,'torch':torch.__version__,'cuda_available':bool(torch.cuda.is_available()),'cuda_count':int(torch.cuda.device_count())}))"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to probe torch runtime via: $PythonExe"
}

$probe = $probeJson | ConvertFrom-Json
Write-Host "[GPU-CHECK] exe=$($probe.exe)"
Write-Host "[GPU-CHECK] torch=$($probe.torch)"
Write-Host "[GPU-CHECK] cuda_available=$($probe.cuda_available), cuda_count=$($probe.cuda_count)"

if (-not $probe.cuda_available -or [int]$probe.cuda_count -lt 1) {
    throw "Strict GPU mode: CUDA is not available in this Python runtime."
}

try {
    $gpuInfo = & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[GPU-CHECK] nvidia-smi:"
        $gpuInfo | ForEach-Object { Write-Host "  $_" }
    }
} catch {
    Write-Host "[GPU-CHECK] nvidia-smi not found; continuing."
}

Write-Host "[GPU-CHECK] PASS"
