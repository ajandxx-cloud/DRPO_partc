Param(
    [string]$PythonExe = "C:/Users/39583/AppData/Local/Programs/Python/Python38/python.exe",
    [int]$Gpu = 0,
    [switch]$OnlySmoke,
    [switch]$AllowSmokeFailure,
    [int]$SmokeEpisodes = 20,
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$CheckScript = Join-Path $PSScriptRoot "gpu_env_check.ps1"
& $CheckScript -PythonExe $PythonExe

$argsList = @(
    "sensitivity_analysis_dspo_plus_spo_oat.py",
    "--profile", "rc_full12",
    "--instance", "RC",
    "--data_seed", "0",
    "--data_seed_test", "1",
    "--python_executable", $PythonExe,
    "--gpu", "$Gpu",
    "--run_smoke_validation",
    "--disable_cache",
    "--primary_metric", "net_profit",
    "--guardrail_quit_delta_pp", "2.0",
    "--guardrail_served_rate_delta", "-0.02",
    "--smoke_episodes", "$SmokeEpisodes"
)

if ($OnlySmoke) {
    $argsList += "--only_smoke"
}

if ($AllowSmokeFailure) {
    $argsList += "--allow_smoke_failure"
}

if ($OutputDir -ne "") {
    $argsList += @("--output_dir", $OutputDir, "--allow_existing_output_dir")
}

Write-Host "[RUN] $PythonExe $($argsList -join ' ')"
& $PythonExe @argsList
if ($LASTEXITCODE -ne 0) {
    throw "Run failed with exit code $LASTEXITCODE"
}

Write-Host "[RUN] Completed successfully."
