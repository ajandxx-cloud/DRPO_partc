Param(
    [string]$PythonExe = "C:/Users/39583/AppData/Local/Programs/Python/Python38/python.exe",
    [int]$Gpu = 0,
    [string]$Instance = "RC",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$CheckScript = Join-Path $PSScriptRoot "gpu_env_check.ps1"
& $CheckScript -PythonExe $PythonExe

$argsList = @(
    "run.py",
    "--algo_name", "DRPO",
    "--instance", $Instance,
    "--gpu", "$Gpu"
)

if ($ExtraArgs) {
    $argsList += $ExtraArgs
}

Write-Host "[RUN] $PythonExe $($argsList -join ' ')"
& $PythonExe @argsList
if ($LASTEXITCODE -ne 0) {
    throw "Training command failed with exit code $LASTEXITCODE"
}

Write-Host "[RUN] Training command finished."
