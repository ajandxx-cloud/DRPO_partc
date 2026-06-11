Param(
    [string]$PythonExe = "C:/Users/39583/AppData/Local/Programs/Python/Python38/python.exe",
    [int]$Gpu = 0,
    [string]$RunPrefix = "SENS_DRPO_OAT_RC_FULL12",
    [string]$FinalOutputDir = "",
    [int]$SaveCount = 1
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$outA = "Experiments/analysis/drpo_sensitivity_oat_rc_full12_shardA_$ts"
$outB = "Experiments/analysis/drpo_sensitivity_oat_rc_full12_shardB_$ts"
if ([string]::IsNullOrWhiteSpace($FinalOutputDir)) {
    $FinalOutputDir = "Experiments/analysis/drpo_sensitivity_oat_rc_full12_parallel_final_$ts"
}

$factorsA = @(
    "outside_option_util",
    "incentive_sens",
    "home_util",
    "k",
    "revenue",
    "fuel_cost"
)
$factorsB = @(
    "home_failure",
    "learning_rate",
    "batch_size",
    "spo_warmup_episodes",
    "spo_rampup_episodes",
    "spo_loss_weight"
)

$commonArgs = @(
    "sensitivity_analysis_dspo_plus_spo_oat.py",
    "--profile", "rc_full12",
    "--instance", "RC",
    "--data_seed", "0",
    "--data_seed_test", "1",
    "--python_executable", $PythonExe,
    "--gpu", "$Gpu",
    "--allow_cache",
    "--skip_existing",
    "--no_smoke_validation",
    "--save_count", "$SaveCount",
    "--primary_metric", "net_profit",
    "--guardrail_quit_delta_pp", "2.0",
    "--guardrail_served_rate_delta", "-0.02",
    "--run_prefix", $RunPrefix
)

$logAOut = "Experiments/analysis/run_rc_full12_shardA_$ts.stdout.log"
$logAErr = "Experiments/analysis/run_rc_full12_shardA_$ts.stderr.log"
$argsA = @($commonArgs + @("--output_dir", $outA, "--factors") + $factorsA)
$pA = Start-Process -FilePath $PythonExe -ArgumentList $argsA -WorkingDirectory $Root -RedirectStandardOutput $logAOut -RedirectStandardError $logAErr -PassThru
Write-Host "[SHARD A] PID=$($pA.Id) OUT=$outA"

$logBOut = "Experiments/analysis/run_rc_full12_shardB_$ts.stdout.log"
$logBErr = "Experiments/analysis/run_rc_full12_shardB_$ts.stderr.log"
$argsB = @($commonArgs + @("--output_dir", $outB, "--factors") + $factorsB)
$pB = Start-Process -FilePath $PythonExe -ArgumentList $argsB -WorkingDirectory $Root -RedirectStandardOutput $logBOut -RedirectStandardError $logBErr -PassThru
Write-Host "[SHARD B] PID=$($pB.Id) OUT=$outB"

$pA.WaitForExit()
$pB.WaitForExit()
if ($pA.ExitCode -ne 0) {
    throw "Shard A failed with exit code $($pA.ExitCode). See $logAErr"
}
if ($pB.ExitCode -ne 0) {
    throw "Shard B failed with exit code $($pB.ExitCode). See $logBErr"
}
Write-Host "[SHARD] Both shards finished."

$finalArgs = @(
    "sensitivity_analysis_dspo_plus_spo_oat.py",
    "--profile", "rc_full12",
    "--instance", "RC",
    "--data_seed", "0",
    "--data_seed_test", "1",
    "--python_executable", $PythonExe,
    "--gpu", "$Gpu",
    "--allow_cache",
    "--skip_existing",
    "--no_smoke_validation",
    "--primary_metric", "net_profit",
    "--guardrail_quit_delta_pp", "2.0",
    "--guardrail_served_rate_delta", "-0.02",
    "--run_prefix", $RunPrefix,
    "--output_dir", $FinalOutputDir
)

Write-Host "[FINAL] Building unified final output: $FinalOutputDir"
& $PythonExe @finalArgs
if ($LASTEXITCODE -ne 0) {
    throw "Final unified pass failed with exit code $LASTEXITCODE"
}

Write-Host "[FINAL] Exporting merged CSV + management insights"
& $PythonExe scripts/export_and_analyze_rc_full12.py `
    --output_dir $FinalOutputDir `
    --export_name rc_full12_all_results_new.csv `
    --insights_name rc_full12_management_insights.csv `
    --summary_name rc_full12_management_summary.txt
if ($LASTEXITCODE -ne 0) {
    throw "Postprocess export failed with exit code $LASTEXITCODE"
}

Write-Host "[DONE] Final output: $FinalOutputDir"
