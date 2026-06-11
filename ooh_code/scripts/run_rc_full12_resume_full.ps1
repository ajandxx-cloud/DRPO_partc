Param(
    [string]$PythonExe = "C:/Users/39583/AppData/Local/Programs/Python/Python38/python.exe",
    [int]$Gpu = 0,
    [string]$RunPrefix = "SENS_DRPO_OAT_RC_FULL12",
    [string]$OutputDir = "",
    [switch]$AllowExistingOutputDir,
    [int]$SaveCount = 1,
    [int]$PersistEveryN = 10,
    [string]$LogFile = ""
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputDir = "Experiments/analysis/drpo_sensitivity_oat_rc_full12_resume_full_$ts"
}

if ([string]::IsNullOrWhiteSpace($LogFile)) {
    $tsLog = Get-Date -Format "yyyyMMdd_HHmmss"
    $LogFile = "Experiments/analysis/run_rc_full12_resume_full_$tsLog.log"
}

$outAbs = Join-Path $Root $OutputDir
if ((Test-Path $outAbs) -and (-not $AllowExistingOutputDir)) {
    $items = Get-ChildItem -Force $outAbs -ErrorAction SilentlyContinue
    if ($items -and $items.Count -gt 0) {
        throw "Output directory already has files: $OutputDir. Re-run with -AllowExistingOutputDir to resume."
    }
}

$argsList = @(
    "sensitivity_analysis_dspo_plus_spo_oat.py",
    "--profile", "rc_full12",
    "--instance", "RC",
    "--data_seed", "0",
    "--data_seed_test", "1",
    "--python_executable", $PythonExe,
    "--gpu", "$Gpu",
    "--allow_cache",
    "--skip_existing",
    "--resume_missing_only",
    "--persist_every_n", "$PersistEveryN",
    "--no_smoke_validation",
    "--save_count", "$SaveCount",
    "--primary_metric", "net_profit",
    "--guardrail_quit_delta_pp", "2.0",
    "--guardrail_served_rate_delta", "-0.02",
    "--run_prefix", $RunPrefix,
    "--output_dir", $OutputDir
)

if ($AllowExistingOutputDir) {
    $argsList += "--allow_existing_output_dir"
}

Write-Host "[RESUME] $PythonExe $($argsList -join ' ')"
Write-Host "[RESUME] Log file: $LogFile"
& $PythonExe @argsList 2>&1 | Tee-Object -FilePath (Join-Path $Root $LogFile)
if ($LASTEXITCODE -ne 0) {
    throw "Resume run failed with exit code $LASTEXITCODE"
}

Write-Host "[RESUME] Completed successfully: $OutputDir"
