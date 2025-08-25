@echo off
set "source_dir=D:\AI_Generated\StabilityMatrix\Data\Models"
set "target_dir_SD=D:\AI_Generated\SD.Next\models"
set "target_dir_Comfy=D:\AI_Generated\ComfyUI-Zluda\models"

for /d %%a in ("%source_dir%\*") do (
  mklink /D "%target_dir_Comfy%\%%~nxa" "%%a"
)
pause