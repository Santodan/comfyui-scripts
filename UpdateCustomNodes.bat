@echo off
setlocal

set "dir=%cd%\custom_nodes"

for /D %%D in ("%dir%\*") do (
    pushd "%%D"
    echo Trying git pull in %%D
    git pull
    popd
)

endlocal
pause