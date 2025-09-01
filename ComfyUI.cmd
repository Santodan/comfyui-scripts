rmdir /s /q "C:\Users\danny\.cache"
pip cache purge
@echo off
echo.
echo "--use-pytorch-cross-attention" , "--use-quad-cross-attention" , "--use-flash-attention" or "--use-sage-attention" ? (1/2/3/4)
set /p Input2=Enter 1 / 2 / 3 / 4:
If /I "%Input2%"=="1" goto py
If /I "%Input2%"=="2" goto quad
If /I "%Input2%"=="3" goto flash
If /I "%Input2%"=="4" goto sage

:: =========== Attention definition ===========
:py
set Attention=--use-pytorch-cross-attention
echo --use-pytorch-cross-attention
goto comfyui
:quad
set %ttention=--use-quad-cross-attention
echo --use-quad-cross-attention
goto comfyui
:flash
set Attention=--use-flash-attention
echo --use-flash-attention
goto comfyui
:sage
set Attention=--use-sage-attention
echo --use-sage-attention
goto comfyui


:: =========== Which ComfyUI? ===========
:comfyui
::set COMMANDLINE_ARGS=--auto-launch --cache-none --disable-smart-memory --reserve-vram 0 %Attention%
set COMMANDLINE_ARGS=--reserve-vram 0.9 --normalvram %Attention% --disable-smart-memory
set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512
set PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

echo ComfyUi-PyTorch or ComfyUI-Zluda? (1/2)
set /p Input=Enter 1 or 2:
If /I "%Input%"=="1" goto pytorch
If /I "%Input%"=="2" goto zluda



:: =========== PyTorch ===========
:pytorch
title ComfyUI PyTorch Launcher
cd /d "D:\AI_Generated\ComfyUI"
cmd /k ComfyUI_Pytorch.bat



:: =========== Which Zluda? ===========
:zluda
title ComfyUI Zluda Launcher
cd /d "D:\AI_Generated\ComfyUI-Zluda"
echo.
echo Zluda-n.bat or custom? (1/2)
set /p Input=Enter 1 or 2:
If /I "%Input%"=="1" goto ZludaN
If /I "%Input%"=="2" goto customcomfy


:: =========== Normal Zluda ===========
:ZludaN
call %~dp0venv\Scripts\activate
pip show numpy | findstr /C:"Version: 1.26.4" >nul
if %errorlevel% neq 0 (
    echo Correct numpy version not found. Reinstalling...
    pip uninstall numpy -y --quiet
    pip install numpy==1.26.4
) else (
    echo Numpy 1.26.4 is already installed.
)
echo Running comfyui-n.bat
cmd /k comfyui-n.bat


:: =========== Custom Zluda ===========
:customcomfy
echo Running comfyui-n_mine.bat
cmd /k comfyui-n_mine.bat