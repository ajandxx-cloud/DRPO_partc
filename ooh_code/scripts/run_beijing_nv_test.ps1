Param(
    [string]$PythonExe = "C:/Users/39583/AppData/Local/Programs/Python/Python38/python.exe",
    [int]$Gpu = 0,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Seeds     = @(40, 67, 97)
$NVehicles = @(15, 40)
$Algos     = @("DSPO", "DRPO")

$CommonArgs = @(
    "--instance",            "Beijing_bus",
    "--max_steps_r",         "240",
    "--max_steps_p",         "0.5",
    "--max_episodes",        "200",
    "--data_seed",           "0",
    "--data_seed_test",      "1",
    "--k",                   "10",
    "--grid_dim",            "11",
    "--veh_capacity",        "12",
    "--incentive_sens",      "-0.25",
    "--home_util",           "1.4",
    "--outside_option_util", "-1.0",
    "--revenue",             "50",
    "--fuel_cost",           "0.6",
    "--home_failure",        "0.1",
    "--batch_size",          "256",
    "--learning_rate",       "0.001",
    "--spo_loss_weight",     "0.7",
    "--spo_warmup_episodes", "5",
    "--spo_rampup_episodes", "10",
    "--save_count",          "1",
    "--log_output",          "file",
    "--debug",               "False",
    "--gpu",                 "$Gpu"
)

$Total   = $Seeds.Count * $NVehicles.Count * $Algos.Count
$Current = 0

Write-Host "=========================================="
Write-Host " Beijing n_vehicles test"
Write-Host " Algos   : $($Algos -join ', ')"
Write-Host " Vehicles: $($NVehicles -join ' vs ')"
Write-Host " Seeds   : $($Seeds -join ', ')"
Write-Host " Total   : $Total runs"
Write-Host "=========================================="

foreach ($nv in $NVehicles) {
    foreach ($algo in $Algos) {
        foreach ($seed in $Seeds) {
            $Current++
            $ExpName = "Beijing_nv${nv}_${algo}_seed${seed}"
            $Suffix  = "_beijing_nv_test"

            $RunArgs = $CommonArgs + @(
                "--algo_name",     $algo,
                "--n_vehicles",    "$nv",
                "--seed",          "$seed",
                "--experiment",    $ExpName,
                "--folder_suffix", $Suffix
            )

            Write-Host ""
            Write-Host "[$Current/$Total] $ExpName"

            if ($DryRun) {
                Write-Host "  [DRY] $PythonExe run.py $($RunArgs -join ' ')"
                continue
            }

            $t0 = Get-Date
            & $PythonExe "run.py" @RunArgs
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "  [ERROR] $ExpName exit=$LASTEXITCODE, skipping"
                continue
            }
            $elapsed = [math]::Round(((Get-Date) - $t0).TotalSeconds, 0)
            Write-Host "  [OK] ${elapsed}s"
        }
    }
}

Write-Host ""
Write-Host "=========================================="
Write-Host " Done. $Total runs."
Write-Host " Results: Experiments/Parcelpoint_py/pricing/*_beijing_nv_test"
Write-Host "=========================================="
