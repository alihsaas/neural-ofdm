@echo off
cd /d "%~dp0"
set CNN=configs/neural_channel_estimation/protocol.json
set ATT=configs/attention_channel_estimation/protocol.json
uv run --extra neural python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || goto fail
uv run --extra neural python scripts/generate_dataset.py --protocol %CNN% || goto fail
uv run --extra neural python scripts/train_channel_estimator.py --protocol %CNN% --skip-existing || goto fail
uv run --extra neural python scripts/evaluate_channel_estimator.py --protocol %CNN% || goto fail
uv run --extra neural python scripts/generate_dataset.py --protocol %ATT% --splits train validation test || goto fail
uv run --extra neural python scripts/train_channel_estimator.py --protocol %ATT% --skip-existing || goto fail
uv run --extra neural python scripts/evaluate_channel_estimator.py --protocol %ATT% || goto fail
echo.
echo Finished. Results are in results\neural_channel_estimation and results\attention_channel_estimation.
pause
exit /b 0
:fail
echo.
echo A step failed. Read the message above.
pause
exit /b 1
