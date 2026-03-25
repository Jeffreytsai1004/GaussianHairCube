@echo off
echo ========================================
echo NexusLauncher Cache Cleaner
echo ========================================
echo.

@REM echo Close NexusLauncher if it is running...
@REM taskkill /f /im pythonw.exe

echo [1/4] Cleaning all __pycache__ folders...
for /d /r %%d in (__pycache__) do @if exist "%%d" (
    echo   Deleting: %%d
    rd /s /q "%%d"
)
echo.

echo [4/4] Cleaning .pyc files...
del /s /q *.pyc 2>nul
echo.

@REM echo Clear old config file
@REM if exist config.json del /f config.json

echo ========================================
echo Cache cleaned successfully!
echo ========================================
pause