$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

$Python = "C:\Users\39583\AppData\Local\Programs\Python\Python38\python.exe"
$OutputDir = Join-Path $Root "Experiments\analysis\yanjiao_drpo_dspo_tuning"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

& $Python `
  "scripts\run_yanjiao_drpo_dspo_tuning.py" `
  "--python_executable" $Python `
  "--gpu" "0" `
  "--episodes" "150" `
  "--output_dir" "Experiments/analysis/yanjiao_drpo_dspo_tuning" `
  "--validation_output_dir" "Experiments/analysis/yanjiao_final_maxprice5_3seed" `
  "--folder_suffix" "_yanjiao_param_tuning_mp5" `
  "--validation_folder_suffix" "_yanjiao_final_mp5_3seed" `
  "--max_retries" "0"
