:: Requirements
:: ============
::
:: Python 3.11:
::   Download from https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
::   IMPORTANT: On the first install screen, make sure to tick "☐ Add python.exe to PATH"
::
:: Git:
::   Download from https://git-scm.com/download/win
::   Use all default options during installation
::
:: Microsoft Visual Studio Compiler:
::   Easiest: Download Build Tools for Visual Studio:
::     https://aka.ms/vs/17/release/vs_BuildTools.exe
::
::   Alternatively: Install Visual Studio (Build Tools only):
::     https://visualstudio.microsoft.com/downloads/
::
::   Ensure the following components are checked:
::     - MSVC
::     - C++ CMake tools for Windows
::     - C++ ATL
::     - C++/CLI Support
::
:: HIP SDK Development (for ROCm 6.5):
::   This should be performed automatically, but if it fails:
::     Download this from google drive https://drive.google.com/file/d/124N7C22-tiC8XR-QmPDqvmfSrF6LRglh/view?usp=sharing
::     Unzip into C:\Program Files\AMD\ROCm\6.5

:: Notes to future self:
:: Updating this script using ChatGPT
:: "Update the URLs for torch, torchaudio, and torchvision, by checking https://github.com/lshqqytiger/TheRock/releases"
:: "Update the URL for triton-3.4.0...cp311-cp311-win_amd64.whl, at https://github.com/lshqqytiger/triton/releases/"

@echo off
setlocal EnableDelayedExpansion

:: Environment Variables YOU MUST SET
:: ==================================
:: The gfxXXXX name of your AMD GPU.  gfx1100 = 7900XTX, gfx1030 = 6800, etc.
set GFX_ARCH=gfx1030 

:: The location of your **new** ComfyUI directory
set COMFYUI_DIR=D:\AI_Generated\ComfyUI

:: No user servicable parts from this point on
:: ===========================================
:: Python and dependencies
:: The location of for HIP SDK (6.5) https://drive.google.com/file/d/124N7C22-tiC8XR-QmPDqvmfSrF6LRglh/view?usp=sharing
set HIP_SDK_DIR=D:\AI_Generated\hip65
set PYTHON_VERSION=3.11

:: Extensions
set SAGE_PATCH_DIR=https://raw.githubusercontent.com/patientx/ComfyUI-Zluda/refs/heads/master/comfy/customzluda/sa
set FLASH_PATCH=https://raw.githubusercontent.com/patientx/ComfyUI-Zluda/master/comfy/customzluda/fa/distributed.py
set "HIP_ZIP_URL=https://nt4.com/amd/HIP-SDK-6.5-develop.zip"
set "HIP_ZIP_PATH=%TEMP%\hip_sdk.zip"


:: Preflight Check: Git
where git >nul 2>&1
set ERRLEVEL=%errorlevel%
if %ERRLEVEL% neq 0 (
    echo "Git is not installed or not in PATH. Install it from https://git-scm.com/download/win."
    exit /b %ERRLEVEL%
)

:: Ensure parent directory of COMFYUI_DIR exists
for %%I in ("%COMFYUI_DIR%") do set "COMFYUI_PARENT=%%~dpI"

if not exist "%COMFYUI_PARENT%" (
    echo Creating parent directory: %COMFYUI_PARENT%
    mkdir "%COMFYUI_PARENT%" >nul 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] Unable to create directory: %COMFYUI_PARENT%
        echo This usually means Administrator privileges are required.
        echo Please do the following manually:
        echo.
        echo    1. Open Command Prompt as Administrator
        echo    2. Run: mkdir "%COMFYUI_PARENT%"
        echo    3. Re-run this script normally
        exit /b 1
    )
)

cd /d "%COMFYUI_PARENT%"
set ERRLEVEL=%errorlevel%
if %ERRLEVEL% neq 0 (
    echo "Failed to change directory to COMFYUI_PARENT."
    exit /b %ERRLEVEL%
)

:: Step 1: Verify Python version
py -%PYTHON_VERSION% -c "import sys; assert sys.version_info >= (3,11), 'Python 3.11 required'"
set ERRLEVEL=%errorlevel%
if %ERRLEVEL% neq 0 (
    echo "Python 3.11 is not installed or not accessible. Download it from https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe and ensure 'Add to PATH' is selected during install."
    exit /b %ERRLEVEL%
)

:: Step 2: Clone ComfyUI
if not exist "%COMFYUI_DIR%" (
    echo Cloning ComfyUI...
    git clone https://github.com/comfyanonymous/ComfyUI.git "%COMFYUI_DIR%"
    set ERRLEVEL=%errorlevel%
    if %ERRLEVEL% neq 0 (
        echo "Failed to clone ComfyUI repository. Ensure %COMFYUI_DIR% does not exist, and that git is installed from https://git-scm.com/download/win."
        exit /b %ERRLEVEL%
    )
) else (
    echo ComfyUI directory already exists, skipping clone of ComfyUI.  Delete %COMFYUI_DIR% if you want to clone ComfyUI.
)

cd /d "%COMFYUI_DIR%"
set ERRLEVEL=%errorlevel%
if %ERRLEVEL% neq 0 (
    echo "Failed to change directory to COMFYUI_DIR."
    exit /b %ERRLEVEL%
)

:: Step 4: Set up virtual environment
py -%PYTHON_VERSION% -m venv venv
set ERRLEVEL=%errorlevel%
if %ERRLEVEL% neq 0 (
    echo "Failed to create virtual environment."
    exit /b %ERRLEVEL%
)

call venv\Scripts\activate
set ERRLEVEL=%errorlevel%
if %ERRLEVEL% neq 0 (
    echo "Failed to activate virtual environment."
    exit /b %ERRLEVEL%
)

:: goto :skip_pip

:: Step 5: Clean previous installations (just in case)
pip uninstall -y torch torchaudio torchvision triton
set ERRLEVEL=%errorlevel%
if %ERRLEVEL% neq 0 (
    echo "Failed to uninstall existing torch/triton packages."
    exit /b %ERRLEVEL%
)

:: Create requirements.txt with all wheel URLs
(
    echo https://github.com/lshqqytiger/TheRock/releases/download/build0/torch-2.7.0+rocm6.5.unofficial-cp311-cp311-win_amd64.whl
    echo https://github.com/lshqqytiger/TheRock/releases/download/build0/torchaudio-2.7.0+rocm6.5.unofficial-cp311-cp311-win_amd64.whl
    echo https://github.com/lshqqytiger/TheRock/releases/download/build0/torchvision-0.22.0+rocm6.5.unofficial-cp311-cp311-win_amd64.whl
    echo https://github.com/lshqqytiger/triton/releases/download/a9c80202/triton-3.4.0+gita9c80202-cp311-cp311-win_amd64.whl
    echo https://raw.githubusercontent.com/patientx/ComfyUI-Zluda/master/comfy/customzluda/fa/flash_attn-2.7.4.post1-py3-none-any.whl
    echo sageattention
) > extra-requirements.txt

pip install -r extra-requirements.txt

:: Step 6: Install ROCm wheels
set ERRLEVEL=%errorlevel%
if %ERRLEVEL% neq 0 (
    echo "Failed to install one or more required wheels."
    exit /b %ERRLEVEL%
)

pip install -r requirements.txt
set ERRLEVEL=%errorlevel%
if %ERRLEVEL% neq 0 (
    echo "Failed to install ComfyUI requirements.txt packages."
    exit /b %ERRLEVEL%
)

pip uninstall numpy -y --quiet
pip install numpy==1.26.4 --quiet

:: Step 10: Patch flash-attn
curl -o "venv\Lib\site-packages\flash_attn\utils\distributed.py" "%FLASH_PATCH%"
set ERRLEVEL=%errorlevel%
if %ERRLEVEL% neq 0 (
    echo "Failed to patch flash_attn distributed.py."
    exit /b %ERRLEVEL%
)

:: Step 11: Patch sage-attention
curl -o "venv\Lib\site-packages\sageattention\attn_qk_int8_per_block.py" "%SAGE_PATCH_DIR%/attn_qk_int8_per_block.py"
set ERRLEVEL=%errorlevel%
if %ERRLEVEL% neq 0 (
    echo "Failed to patch sageattention patch #1"
    exit /b %ERRLEVEL%
)
curl -o "venv\Lib\site-packages\sageattention\attn_qk_int8_per_block_causal.py" "%SAGE_PATCH_DIR%/attn_qk_int8_per_block_causal.py"
set ERRLEVEL=%errorlevel%
if %ERRLEVEL% neq 0 (
    echo "Failed to patch sageattention patch #2"
    exit /b %ERRLEVEL%
)
curl -o "venv\Lib\site-packages\sageattention\quant_per_block.py" "%SAGE_PATCH_DIR%/quant_per_block.py"
set ERRLEVEL=%errorlevel%
if %ERRLEVEL% neq 0 (
    echo "Failed to patch sageattention patch #3"
    exit /b %ERRLEVEL%
)

:: Step 12: Patch for Triton
pushd .
cd venv\Lib\site-packages
set ERRLEVEL=%errorlevel%
if %ERRLEVEL% neq 0 (
    echo "Failed to change directory to venv\Lib\site-packages."
    exit /b %ERRLEVEL%
)
curl -L -o torch-for-triton-a9c80202-hacks.tar https://nt4.com/torch-for-triton-a9c80202-hacks.tar
set ERRLEVEL=%errorlevel%
if %ERRLEVEL% neq 0 (
    echo "Failed to download Triton torch compatibility patch."
    exit /b %ERRLEVEL%
)
tar -xvf torch-for-triton-a9c80202-hacks.tar
set ERRLEVEL=%errorlevel%
if %ERRLEVEL% neq 0 (
    echo "Failed to extract Triton patch archive."
    exit /b %ERRLEVEL%
)
popd

:: Step 13: Copy Python libs
xcopy /E /I /Y "%LocalAppData%\Programs\Python\Python311\libs" "venv\libs"
set ERRLEVEL=%errorlevel%
if %ERRLEVEL% neq 0 (
    echo "Failed to copy Python311\libs to virtual environment."
    exit /b %ERRLEVEL%
)

:skip_pip

:: Check if HIP SDK exists, otherwise download and extract it
set "HIP_SDK_PARENT=%HIP_SDK_DIR%"
for %%A in ("%HIP_SDK_PARENT%\.") do set "HIP_SDK_PARENT=%%~dpA"
if "%HIP_SDK_PARENT:~-1%"=="\" set "HIP_SDK_PARENT=%HIP_SDK_PARENT:~0,-1%"
set "TEST_FILE=%HIP_SDK_PARENT%\__write_test__.tmp"
echo HIP_SDK_DIR: %HIP_SDK_DIR%
echo HIP_SDK_PARENT: %HIP_SDK_PARENT%
echo HIP_ZIP_PATH: %HIP_ZIP_PATH%
if not exist "%HIP_SDK_DIR%\bin\clang.exe" (
    echo HIP SDK not found at %HIP_SDK_DIR%.
    if exist "%HIP_ZIP_PATH%" (
        echo Using previously downloaded file: %HIP_ZIP_PATH%
    ) else (
        echo Attempting to download...
        curl -o "%HIP_ZIP_PATH%" "%HIP_ZIP_URL%"
        set ERRLEVEL=!errorlevel!
        if !ERRLEVEL! neq 0 (
            echo "Failed to download HIP SDK archive."
            exit /b !ERRLEVEL!
        )
        if not exist "%HIP_ZIP_PATH%" (
            echo "HIP SDK archive not found after download attempt."
            exit /b 1
        )
    )

    :: --- Check write permission to HIP_SDK_PARENT ---
    :check_hip_write
    echo Checking write access to: %HIP_SDK_PARENT%

    :: Create test file
    echo Checking to see if we have write access to the SDK directory by writing to %TEST_FILE%
    break > "%TEST_FILE%" 2>nul
    set ERRLEVEL=%errorlevel%

    if %ERRLEVEL% neq 0 (
        echo We couldn't write to %TEST_FILE%
    )

    :: Try modifying it
    if exist "%TEST_FILE%" (
        >> "%TEST_FILE%" echo test >>nul 2>nul
        set MODIFY_ERR=!errorlevel!
        del "%TEST_FILE%" >nul 2>&1
    )

    :: Check create permission
    if %ERRLEVEL% neq 0 (
        echo [ERROR] Could not create test file in %HIP_SDK_PARENT%.
        goto :request_admin_fix
    )

    :: Check modify permission
    if defined MODIFY_ERR (
        if !MODIFY_ERR! neq 0 (
            echo [ERROR] Could not write to test file in %HIP_SDK_PARENT%.
            goto :request_admin_fix
        )
    )

    echo Write access confirmed.
    goto :hip_access_ok

    :request_admin_fix
    echo.
    echo You do not have write access to:
    echo     %HIP_SDK_PARENT%
    echo.
    echo To fix this, open an **Admin Command Prompt** and run:
    echo     takeown /f "%HIP_SDK_PARENT%" /r /d y
    echo     icacls "%HIP_SDK_PARENT%" /grant "%USERNAME%:F" /t
    echo.
    pause
    goto check_hip_write

    :hip_access_ok
    :: --- Extract the HIP SDK using tar ---
    echo Extracting HIP SDK archive using tar...  tar -xf "%HIP_ZIP_PATH%" -C "%HIP_SDK_PARENT%"
    tar -xf "%HIP_ZIP_PATH%" -C "%HIP_SDK_PARENT%"
    set ERRLEVEL=%errorlevel%
    if not %ERRLEVEL%==0 (
        echo [ERROR] Failed to extract HIP SDK archive using tar.
        exit /b %ERRLEVEL%
    )

    if not exist "%HIP_SDK_DIR%\bin\clang.exe" (
        echo HIP SDK not found at %HIP_SDK_DIR%, which is annoying, since we just put it there.
        exit /b 1
    )
)


:: Step 14: Create run script
set RUN_SCRIPT=%COMFYUI_DIR%\run_comfy_rocm.bat
echo @echo off > "%RUN_SCRIPT%"
set ERRLEVEL=%errorlevel%
if %ERRLEVEL% neq 0 (
    echo "Failed to generate run script."
    exit /b %ERRLEVEL%
)
echo cd /d "%COMFYUI_DIR%" >> "%RUN_SCRIPT%"
echo call venv\Scripts\activate >> "%RUN_SCRIPT%"
echo set HIP_PATH=%HIP_SDK_DIR% >> "%RUN_SCRIPT%"
echo set TRITON_OVERRIDE_ARCH=%GFX_ARCH% >> "%RUN_SCRIPT%"
echo set COMFYUI_PATH=%COMFYUI_DIR% >> "%RUN_SCRIPT%"
echo set COMFYUI_MODEL_PATH=%COMFYUI_DIR%\models >> "%RUN_SCRIPT%"
echo set CC=%HIP_SDK_DIR%\bin\clang.exe >> "%RUN_SCRIPT%"
echo set CXX=%HIP_SDK_DIR%\bin\clang++.exe >> "%RUN_SCRIPT%"
echo set CXXFLAGS=-march=native -mtune=native >> "%RUN_SCRIPT%"
echo set FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE >> "%RUN_SCRIPT%"
echo set MIOPEN_FIND_MODE=2 >> "%RUN_SCRIPT%"
echo set MIOPEN_LOG_LEVEL=3 >> "%RUN_SCRIPT%"
echo set MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD=0 >> "%RUN_SCRIPT%"
echo set PATH=%HIP_SDK_DIR%\bin;%%PATH%% >> "%RUN_SCRIPT%"
echo git pull >> "%RUN_SCRIPT%"
echo pip install -U comfyui-frontend-package comfyui-workflow-templates av comfyui-embedded-docs --quiet >> "%RUN_SCRIPT%"
echo python main.py --use-quad-cross-attention >> "%RUN_SCRIPT%"

:: Step 15: Final Message
echo.
echo Done.
echo You can now launch ComfyUI using patientx's launch script (actually that doesn't seem to be working very well, but run it once because it sets up the front end libraries:
echo     %RUN_SCRIPT%
echo The you'll have to download a more complex script (which may require a tiny bit of modification if you don't have a gfx1100 or used unusual paths) from here:
echo     https://gist.github.com/sfinktah/85459b3a9bcf959d6c3ace7e777cb66e#file-comfy2-bat
echo 
pause

exit /b 0

:: Helper function to check errors with reason
:checkError
set ERRLEVEL=%1
shift
set MSG=%*
if not %ERRLEVEL%==0 (
    echo [ERROR] %MSG%
    exit /b %ERRLEVEL%
)
exit /b 0
