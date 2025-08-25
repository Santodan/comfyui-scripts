@echo off
echo === Cleaning ZLUDA, MIOpen, and Triton cache ===

echo [1/3] Cleaning ZLUDA cache...
rmdir /S /Q "%LOCALAPPDATA%\ZLUDA"

echo [2/3] Cleaning MIOpen cache...
rmdir /S /Q "%USERPROFILE%\.miopen"

echo [3/3] Cleaning Triton cache...
rmdir /S /Q "%USERPROFILE%\.triton"

echo === Done. All caches cleaned. ===
pause
