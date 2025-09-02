@ECHO OFF
cls
netsh interface set interface "Ethernet 2" disable
netsh interface set interface "Ethernet 2" enable
pause