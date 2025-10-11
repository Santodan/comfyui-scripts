@echo off
setlocal

set MAIN_DIR=D:\AI_Generated\ComfyUI

:: Wildcards
Echo === Symlink Wildcards ===
set "source_dir=D:\AI_Generated\Wildcards"
set "target_dir=%MAIN_DIR%\Wildcards"
if exist "%target_dir%" (
    echo Deleting existing %target_dir%
    rmdir /S /Q "%target_dir%"
)
mklink /D "%target_dir%" "%source_dir%"

:: Models
Echo === Symlink Models ===
set "source_dir=D:\AI_Generated\comfyui_models"
set "target_dir=%MAIN_DIR%\models"
if exist "%target_dir%" (
    echo Deleting existing %target_dir%
    rmdir /S /Q "%target_dir%"
)
mklink /D "%target_dir%" "%source_dir%"

:: Output
Echo === Symlink Output ===
set "source_dir=D:\AI_Generated\Generated\ComfyUI"
set "target_dir=%MAIN_DIR%\output"
if exist "%target_dir%" (
    echo Deleting existing %target_dir%
    rmdir /S /Q "%target_dir%"
)
mklink /D "%target_dir%" "%source_dir%"

:: Workflow
Echo === Symlink Workflow ===
set "source_dir=D:\AI_Generated\Workflows"
set "target_dir=%MAIN_DIR%\user\default\workflows"
:: Ensure parent folder exists
for %%A in ("%target_dir%") do set "parent_dir=%%~dpA"
if not exist "%parent_dir%" (
    echo Creating missing parent directories...
    mkdir "%parent_dir%"
)
if exist "%target_dir%" (
    echo Deleting existing %target_dir%
    rmdir /S /Q "%target_dir%"
)
mklink /D "%target_dir%" "%source_dir%"

pause
