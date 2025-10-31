@echo off

:: Activate python venv
call %~dp0venv\Scripts\activate


:: =========== Run Once Per Day Task - Update Custom Nodes ===========
setlocal enabledelayedexpansion
set "LastRunFile=%~dp0last_run.txt"

:: Get today in stable YYYY-MM-DD format
for /f %%a in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set "Today=%%a"

:: Read last run date safely
if exist "%LastRunFile%" (
    for /f "usebackq tokens=* delims=" %%A in ("%LastRunFile%") do set "LastRun=%%A"
    :: Force strip to first 10 chars to kill CR/LF
    set "LastRun=!LastRun:~0,10!"
) else (
    set "LastRun="
)

echo Today   =[%Today%]
echo LastRun=[%LastRun%]

if "!LastRun!"=="%Today%" (
    echo Daily script already ran today. Skipping...
) else (
    echo Running daily script...
    call %~dp0UpdateCustomNodes.bat
    >"%LastRunFile%" echo %Today%
)

:: Get torch version
set "TORCH_VERSION="
for /f "tokens=2 delims=: " %%A in ('pip show torch 2^>nul ^| findstr /R "^Version:"') do set "TORCH_VERSION=%%A"

if not defined TORCH_VERSION (
  set "TORCH_VERSION=NO_TORCH"
)

echo Torch version: %TORCH_VERSION%

:: Correct numpy
pip show numpy | findstr /C:"Version: 1.26.4" >nul
if %errorlevel% neq 0 (
    echo Correct numpy version not found. Reinstalling...
    pip uninstall numpy -y --quiet
    pip install numpy==1.26.4
) else (
    echo Numpy 1.26.4 is already installed.
)
set "NUMPY_VERSION="
for /f "tokens=2 delims=: " %%A in ('pip show numpy 2^>nul ^| findstr /R "^Version:"') do set "NUMPY_VERSION=%%A"

if not defined NUMPY_VERSION (
  set "NUMPY_VERSION=NO_NUMPY"
)

::echo Numpy version: %NUMPY_VERSION%

REM  if not defined TRITON_OVERRIDE_ARCH set TRITON_OVERRIDE_ARCH=gfx1100
:: this is for rx6800 (my gpu) , find yours and change if necessary
set TRITON_OVERRIDE_ARCH=gfx1030
echo Triton Architecture: %TRITON_OVERRIDE_ARCH%
set ROCM_VERSION=6.5
set TRITON_CACHE_DIR=%~dp0.triton-%TORCH_VERSION%-%ROCM_VERSION%-%NUMPY_VERSION%-%TEST_FACTOR%
set TORCHINDUCTOR_CACHE_DIR=%~dp0.inductor-%TORCH_VERSION%-%ROCM_VERSION%-%NUMPY_VERSION%-%TEST_FACTOR%
set NUMBA_CACHE_DIR=%~dp0.numba-%TORCH_VERSION%-%ROCM_VERSION%-%NUMPY_VERSION%-%TEST_FACTOR%
set ZLUDA_CACHE_DIR=%~dp0.zluda-%TORCH_VERSION%-%ROCM_VERSION%-%NUMPY_VERSION%-%TEST_FACTOR%
:: these don't appear to do anything
set MIOPEN_CACHE_DIR=%~dp0.miopen-%TORCH_VERSION%-%ROCM_VERSION%-%NUMPY_VERSION%-%TEST_FACTOR%
set DEFAULT_CACHE_DIR=%~dp0.default-cache-%TORCH_VERSION%-%ROCM_VERSION%-%NUMPY_VERSION%-%TEST_FACTOR%
set CUPY_CACHE_DIR=%~dp0.cupy-%TORCH_VERSION%-%ROCM_VERSION%-%NUMPY_VERSION%-%TEST_FACTOR%

set COMFYUI_PATH=%~dp0
set COMFYUI_MODEL_PATH=%~dp0models
:: set "REAL_ZLUDA_CACHE=%USERPROFILE%\AppData\Local\ZLUDA"

:: force miopen to use a local cache (if it isn't sfinktah's fantastic zluda)
set "REAL_MIOPEN_CACHE=%USERPROFILE%\.miopen"
echo Forcing miopen cache...
rmdir /q /s %REAL_MIOPEN_CACHE%
mkdir %MIOPEN_CACHE_DIR%
mklink /D %REAL_MIOPEN_CACHE% %MIOPEN_CACHE_DIR%

:: change comfyui path to yours on both
::set COMFYUI_PATH=D:\AI_Generated\ComfyUI
::set COMFYUI_MODEL_PATH=%COMFYUI_PATH%\models
::set TRITON_CACHE_DIR=%COMFYUI_PATH%\.triton_cache
::set MIOPEN_USER_DB_PATH=%COMFYUI_PATH%\.miopen_cache
::set ZLUDA_CACHE_DIR=%COMFYUI_PATH%\.zluda_cache


::cd /d %COMFYUI_PATH%
::call %COMFYUI_PATH%\venv\Scripts\activate 

:: change hip_path to the custom hip you downloaded and ectracted
set HIP_PATH=D:\AI_Generated\hip65
set CC=%HIP_PATH%\bin\clang.exe
set CXX=%HIP_PATH%\bin\clang++.exe
set CXXFLAGS=-march=native -mtune=native

:: performance?
set TF_ENABLE_ONEDNN_OPTS=0

set PYTHON="%~dp0/venv/Scripts/python.exe"
set GIT=
set VENV_DIR=./venv

set FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
set FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=TRUE
set MIOPEN_FIND_MODE=2
set MIOPEN_LOG_LEVEL=3
::set MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD=0


::set COMMANDLINE_ARGS=--auto-launch --reserve-vram 0.9 --normalvram --use-quad-cross-attention
:: "--use-pytorch-cross-attention" , "--use-quad-cross-attention" , "--use-flash-attention" or "--use-sage-attention" 

:: Add directories to PATH for this session only (change to your paths, for the 65 ones)
set PATH=D:\AI_Generated\hip65\rocm;D:\AI_Generated\hip65\bin;D:\AI_Generated\hip65\cmake;D:\AI_Generated\hip65\include;D:\AI_Generated\hip65\lib;%COMFYUI_PATH%;%COMFYUI_MODEL_PATH%;%PATH%

git pull

:: silently check and update packages
pip install -U comfyui-frontend-package==1.28.7 comfyui-workflow-templates av comfyui-embedded-docs --quiet

python main.py %COMMANDLINE_ARGS%
