@echo off

set FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
set FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=TRUE

set MIOPEN_FIND_MODE=2
set MIOPEN_LOG_LEVEL=3

set PYTHON="%~dp0/venv/Scripts/python.exe"
set GIT=
set VENV_DIR=./venv

set TRITON_OVERRIDE_ARCH=gfx1030
::set SAGE_BM_SIZE=64
::set SAGE_BN_SIZE=16
::set SAGE_ATTENTION_BLOCK_M=64
::set SAGE_ATTENTION_BLOCK_N=16
::set SAGE_ATTENTION_NUM_WARPS=4
::set SAGE_ATTENTION_NUM_STAGES={1,4}
::set SAGE_ATTENTION_STAGE=1
::set SAGE_ATTENTION_WAVES_PER_EU={3,4}
set TRITON_CACHE_DIR=%~dp0/.triton
set MIOPEN_CACHE_DIR=%~dp0/.miopen
set ZLUDA_CACHE_DIR=%~dp0/.zluda_cache

::set COMMANDLINE_ARGS=--auto-launch --use-pytorch-cross-attention --reserve-vram 0.9 --normalvram
::set COMMANDLINE_ARGS=--auto-launch --use-sage-attention --cache-none --disable-smart-memory --reserve-vram 0
set COMMANDLINE_ARGS=--auto-launch --use-sage-attention --disable-smart-memory --reserve-vram 0.9
:: "--use-pytorch-cross-attention" , "--use-quad-cross-attention" , "--use-flash-attention" or "--use-sage-attention" 
:: --fp16-vae
:: --lowvram (630s) --highvram (OOM) --normalvram (384s) --gpu-only
:: --auto-launch --disable-auto-launch
::  --help
:: more options https://www.reddit.com/r/comfyui/comments/15jxydu/comfyui_command_line_arguments_informational/

:: --fast --cache-lru 10

:: GPU Restart
:: pnputil /enum-devices /class Display
:: pnputil /restart-device "PCI\VEN_1002&DEV_73BF&SUBSYS_0E3A1002&REV_C3\6&818bbad&0&00000019"

::   --verbose [{DEBUG,INFO,WARNING,ERROR,CRITICAL}]
::                        Set the logging level
::  --log-stdout          Send normal process output to stdout instead of stderr (default).

echo *** Checking numpy version
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

set ZLUDA_COMGR_LOG_LEVEL=1

echo *** Checking and updating to new version if possible 
copy comfy\customzluda\zluda-default.py comfy\zluda.py /y >NUL
git pull
copy comfy\customzluda\zluda.py comfy\zluda.py /y >NUL
echo.
.\zluda\zluda.exe -- %PYTHON% main.py %COMMANDLINE_ARGS%
pause
