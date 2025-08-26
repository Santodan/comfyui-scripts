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
:comfyui
::set COMMANDLINE_ARGS=--auto-launch --cache-none --disable-smart-memory --reserve-vram 0 %Attention%
set COMMANDLINE_ARGS=--reserve-vram 0.9 --normalvram %Attention% --disable-smart-memory
set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512
::set PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

echo ComfyUi-PyTorch or ComfyUI-Zluda? (1/2)
set /p Input=Enter 1 or 2:
If /I "%Input%"=="1" goto pytorch
If /I "%Input%"=="2" goto zluda
:pytorch
cd /d "D:\AI_Generated\ComfyUI"
call venv\Scripts\activate.bat
pip show numpy | findstr /C:"Version: 1.26.4" >nul
if %errorlevel% neq 0 (
    echo Correct numpy version not found. Reinstalling...
    pip uninstall numpy -y --quiet
    pip install numpy==1.26.4
) else (
    echo Numpy 1.26.4 is already installed.
)
call venv\Scripts\deactivate.bat
cmd /k ComfyUI_Pytorch.bat
:zluda
title ComfyUI Zluda Launcher
cd /d "D:\AI_Generated\ComfyUI-Zluda"

set COMFYUI_PATH=D:\AI_Generated\ComfyUI-Zluda
set TRITON_CACHE_DIR=%COMFYUI_PATH%\.triton_cache 
set MIOPEN_USER_DB_PATH=%COMFYUI_PATH%\.miopen_cache 
set ZLUDA_CACHE_DIR=%COMFYUI_PATH%\.zluda_cache 

echo.
echo Zluda-n.bat or custom? (1/2)
set /p Input=Enter 1 or 2:
If /I "%Input%"=="1" goto ZludaN
If /I "%Input%"=="2" goto customcomfy
:ZludaN
:: and to check it all works and to tune it
set TRITON_PRINT_AUTOTUNING=1
set TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1
echo Zluda-n.bat
cmd /k comfyui-n.bat
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
:customcomfy
    ::echo Setting environment variables...
    set FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
    set FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=TRUE

    set MIOPEN_FIND_MODE=2
    set MIOPEN_LOG_LEVEL=3

    ::set AMD_LOG_LEVEL=1
    ::set DISABLE_ADDMM_CUDA_LT=1

    @REM set PYTHON="%~dp0/venv/Scripts/python.exe"
    @REM set PYTHON="D:\AI_Generated\ComfyUI-Zluda\venv\Scripts\python.exe"
    set PYTHON=".\venv\Scripts\python.exe"
    @REM echo *** Using Python: %PYTHON%
    set GIT=
    set VENV_DIR=./venv

    ::set COMMANDLINE_ARGS=--auto-launch --reserve-vram 0.9 --normalvram %Attention%
	::set COMMANDLINE_ARGS=--auto-launch --cache-none --disable-smart-memory --reserve-vram 0 %Attention%
    :: "--use-pytorch-cross-attention" , "--use-quad-cross-attention" , "--use-flash-attention" or "--use-sage-attention" 
    :: --fp16-vae
    :: --lowvram (630s) --highvram (OOM) --normalvram (384s) --gpu-only
    :: --auto-launch --disable-auto-launch
    ::  --help
	:: --disable-all-custom-nodes
    :: more options https://www.reddit.com/r/comfyui/comments/15jxydu/comfyui_command_line_arguments_informational/

    :: --fast --cache-lru 10

    :: GPU Restart
    :: pnputil /enum-devices /class Display
    :: pnputil /restart-device "PCI\VEN_1002&DEV_73BF&SUBSYS_0E3A1002&REV_C3\6&818bbad&0&00000019"

    ::   --verbose [{DEBUG,INFO,WARNING,ERROR,CRITICAL}]
    ::                        Set the logging level
    ::  --log-stdout          Send normal process output to stdout instead of stderr (default).

    set ZLUDA_COMGR_LOG_LEVEL=1

    echo *** Checking and updating to new version if possible 
    copy comfy\customzluda\zluda-default.py comfy\zluda.py /y >NUL
    git pull
    copy comfy\customzluda\zluda.py comfy\zluda.py /y >NUL
    echo.

    echo Launching ComfyUI with Zluda...
    .\zluda\zluda.exe -- %PYTHON% main.py %COMMANDLINE_ARGS%
    pause
