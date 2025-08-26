@echo off
setlocal

::git clone
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI

:: === Configure log file ===
set MAIN_DIR=D:\AI_Generated\ComfyUI
set LOGFILE=%MAIN_DIR%\setup_log.txt
set PYTHON_EXE=python

:: === Your original script starts below ===

:: Set paths
set VENV_DIR=%MAIN_DIR%\venv

:: Step 1: Create virtual environment
cd /d %MAIN_DIR%
%PYTHON_EXE% -m venv venv

:: Step 2: Activate the environment
call %VENV_DIR%\Scripts\activate.bat

:: Step 3: Install base requirements
pip install -r requirements.txt

:: Step 4: Remove default torch packages
pip uninstall -y torch torchaudio torchvision

:: Step 5: Install custom ROCm Torch wheels (must be downloaded beforehand)
pip install torch-2.7.0+rocm6.5.unofficial-cp311-cp311-win_amd64.whl
pip install torchaudio-2.7.0+rocm6.5.unofficial-cp311-cp311-win_amd64.whl
pip install torchvision-0.22.0+rocm6.5.unofficial-cp311-cp311-win_amd64.whl

:: Step 6: Install Triton
pip install https://github.com/lshqqytiger/triton/releases/download/a9c80202/triton-3.4.0+gita9c80202-cp311-cp311-win_amd64.whl

:: Step 7: Install Flash Attention
pip install https://github.com/patientx/ComfyUI-Zluda/raw/master/comfy/customzluda/fa/flash_attn-2.7.4.post1-py3-none-any.whl

:: Step 8: Install SageAttention
pip install sageattention

:: numpy 1.26.4
pip uninstall numpy -y --quiet
pip install numpy==1.26.4

:: Step 9: Replace distributed.py in flash_attn
curl -L -o %VENV_DIR%\Lib\site-packages\flash_attn\utils\distributed.py https://raw.githubusercontent.com/patientx/ComfyUI-Zluda/master/comfy/customzluda/fa/distributed.py

:: Step 10: Overwrite files in sageattention
curl -L -o %VENV_DIR%\Lib\site-packages\sageattention\attn_qk_int8_per_block.py https://raw.githubusercontent.com/patientx/ComfyUI-Zluda/refs/heads/master/comfy/customzluda/sa/attn_qk_int8_per_block.py
curl -L -o %VENV_DIR%\Lib\site-packages\sageattention\attn_qk_int8_per_block_causal.py https://raw.githubusercontent.com/patientx/ComfyUI-Zluda/refs/heads/master/comfy/customzluda/sa/attn_qk_int8_per_block_causal.py
curl -L -o %VENV_DIR%\Lib\site-packages\sageattention\_quant_per_block.py https://raw.githubusercontent.com/patientx/ComfyUI-Zluda/refs/heads/master/comfy/customzluda/sa/quant_per_block.py

:: Step 11: Copy Python libs folder
xcopy /E /I /Y "C:\Users\danny\AppData\Local\Programs\Python\Python311\libs" "%VENV_DIR%\libs"

:: Step 12: Overwrite torch folder with hacks
curl -L -o "%MAIN_DIR%\torch_hacks.tar" https://nt4.com/torch-for-triton-a9c80202-hacks.tar

:: Extract .tar file into temp folder and copy contents into torch folder
mkdir "%MAIN_DIR%\torch_hacks"
tar -xf "%MAIN_DIR%\torch_hacks.tar" -C "%MAIN_DIR%\torch_hacks"
xcopy /E /I /Y "%MAIN_DIR%\torch_hacks" "%VENV_DIR%\Lib\site-packages\torch"

echo Setup complete!
pause
exit /b
