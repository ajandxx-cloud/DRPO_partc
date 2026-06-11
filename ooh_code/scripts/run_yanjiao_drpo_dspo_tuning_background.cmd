@echo off
setlocal
cd /d "%~dp0\.."
set PY=C:\Users\39583\AppData\Local\Programs\Python\Python38\python.exe
set OUT=Experiments\analysis\yanjiao_drpo_dspo_tuning_run3
if not exist "%OUT%" mkdir "%OUT%"
echo Started at %DATE% %TIME% > "%OUT%\runner_started.txt"
echo Wrapper cwd: %CD% >> "%OUT%\runner_started.txt"
echo Python: %PY% >> "%OUT%\runner_started.txt"
echo Launching tuning script... >> "%OUT%\runner_started.txt"
"%PY%" scripts\run_yanjiao_drpo_dspo_tuning.py ^
  --python_executable "%PY%" ^
  --gpu 0 ^
  --episodes 150 ^
  --output_dir Experiments/analysis/yanjiao_drpo_dspo_tuning_run3 ^
  --validation_output_dir Experiments/analysis/yanjiao_final_maxprice5_3seed_run3 ^
  --folder_suffix _yanjiao_param_tuning_mp5_run3 ^
  --validation_folder_suffix _yanjiao_final_mp5_3seed_run3 ^
  --max_retries 0 ^
  > "%OUT%\runner_stdout.log" 2> "%OUT%\runner_stderr.log"
set EXITCODE=%ERRORLEVEL%
echo Finished at %DATE% %TIME% with exit code %EXITCODE% >> "%OUT%\runner_started.txt"
exit /b %EXITCODE%
