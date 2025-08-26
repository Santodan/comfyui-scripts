@echo off
:: change comfyui path to yours on both

:: change comfyui path to yours on both
set COMFYUI_PATH=D:\AI_Generated\ComfyUI
set COMFYUI_MODEL_PATH=%COMFYUI_PATH%\models
set TRITON_CACHE_DIR=%COMFYUI_PATH%\.triton_cache
set MIOPEN_USER_DB_PATH=%COMFYUI_PATH%\.miopen_cache
set ZLUDA_CACHE_DIR=%COMFYUI_PATH%\.zluda_cache


cd /d %COMFYUI_PATH%
call %COMFYUI_PATH%\venv\Scripts\activate 

:: change hip_path to the custom hip you downloaded and ectracted
set HIP_PATH=D:\AI_Generated\hip65
set CC=%HIP_PATH%\bin\clang.exe
set CXX=%HIP_PATH%\bin\clang++.exe
set CXXFLAGS=-march=native -mtune=native

:: this is for rx6800 (my gpu) , find yours and change if necessary
set TRITON_OVERRIDE_ARCH=gfx1030

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
pip install -U comfyui-frontend-package comfyui-workflow-templates av comfyui-embedded-docs --quiet
python main.py %COMMANDLINE_ARGS%
