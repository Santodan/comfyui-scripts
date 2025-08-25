@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

REM Activate virtual environment
call "%~dp0venv\Scripts\activate.bat"
if errorlevel 1 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

echo Cleaning up compiled Python files...

REM Delete __pycache__ folders
FOR /D /R %%F IN (__pycache__) DO (
    rmdir /S /Q "%%F" >NUL 2>NUL
)

REM Delete .pyc and .pyo files
FOR /R %%F IN (*.pyc *.pyo) DO (
    del /F /Q "%%F" >NUL 2>NUL
)

echo.
echo Changing directory to custom_nodes...
pushd "%~dp0\custom_nodes"
if errorlevel 1 (
    echo Failed to enter custom_nodes folder.
    pause
    exit /b 1
)

echo Searching for requirements.txt files in custom_nodes subfolders...

FOR /D %%D IN (*) DO (
    IF EXIST "%%D\requirements.txt" (
        echo ----------------------------------------------------
        echo Found: %%D\requirements.txt
        echo Installing requirements in: %%D
        pushd "%%D"
        pip install -r requirements.txt
        popd
    )
)

popd

echo.
echo Checking numpy version...
pip show numpy | findstr /C:"Version: 1.26.4" >nul
if errorlevel 1 (
    echo Correct numpy version not found. Reinstalling...
    pip uninstall numpy -y --quiet
    pip install numpy==1.26.4
) else (
    echo Numpy 1.26.4 is already installed.
)

echo.
echo All done.
pause
