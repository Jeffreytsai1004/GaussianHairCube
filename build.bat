@echo off
chcp 65001 >nul 2>nul
REM ============================================
REM GaussianHairCube Build Script
REM ============================================
REM Usage:
REM   build.bat           - Default build (directory mode)
REM   build.bat onefile   - Single file mode
REM   build.bat clean     - Clean build directories
REM   build.bat debug     - Debug mode (with console)
REM   build.bat help      - Show help
REM ============================================

setlocal enabledelayedexpansion

REM Set title
title GaussianHairCube Build Script

REM Get script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Set virtual environment path
set "VENV_DIR=%SCRIPT_DIR%venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"
set "VENV_PYINSTALLER=%VENV_DIR%\Scripts\pyinstaller.exe"

REM Check arguments
if "%1"=="help" goto :help
if "%1"=="--help" goto :help
if "%1"=="-h" goto :help
if "%1"=="clean" goto :clean
if "%1"=="onefile" goto :onefile
if "%1"=="debug" goto :debug

REM Default build
goto :build_onedir

:help
echo.
echo ============================================
echo   GaussianHairCube Build Script
echo ============================================
echo.
echo Usage:
echo   build.bat           Default build (directory mode, recommended)
echo   build.bat onefile   Single file mode (slower startup)
echo   build.bat clean     Clean build directories
echo   build.bat debug     Debug mode (with console window)
echo   build.bat help      Show this help
echo.
echo Output:
echo   Directory mode: dist\GaussianHairCube\GaussianHairCube.exe
echo   Single file mode: dist\GaussianHairCube.exe
echo.
echo Virtual Environment:
echo   Uses venv folder in current directory
echo.
goto :end

:clean
echo.
echo Cleaning build directories...
echo.

if exist build (
    echo Removing build\ ...
    rmdir /s /q build
)

if exist dist (
    echo Removing dist\ ...
    rmdir /s /q dist
)

if exist __pycache__ (
    echo Removing __pycache__\ ...
    rmdir /s /q __pycache__
)

REM Clean __pycache__ in subdirectories
for /d /r . %%d in (__pycache__) do (
    if exist "%%d" (
        echo Removing %%d ...
        rmdir /s /q "%%d"
    )
)

echo.
echo Clean complete!
echo.
goto :end

:check_venv
REM Check if virtual environment exists
if not exist "%VENV_DIR%" (
    echo ERROR: Virtual environment not found!
    echo.
    echo Please create virtual environment first:
    echo   python -m venv venv
    echo   venv\Scripts\activate
    echo   pip install -r requirements.txt
    echo.
    exit /b 1
)

if not exist "%VENV_PYTHON%" (
    echo ERROR: Virtual environment Python not found: %VENV_PYTHON%
    exit /b 1
)

echo Using virtual environment: %VENV_DIR%
echo.
exit /b 0

:check_pyinstaller
REM Check if PyInstaller is installed in virtual environment
"%VENV_PYTHON%" -c "import PyInstaller" >nul 2>nul
if %errorlevel% neq 0 (
    echo PyInstaller not installed, installing...
    "%VENV_PIP%" install pyinstaller
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install PyInstaller
        exit /b 1
    )
)
exit /b 0

:build_onedir
echo.
echo ============================================
echo   Building GaussianHairCube (Directory Mode)
echo ============================================
echo.

REM Check virtual environment
call :check_venv
if %errorlevel% neq 0 goto :error

REM Check PyInstaller
call :check_pyinstaller
if %errorlevel% neq 0 goto :error

REM Check icon file
if not exist "assets\icon.ico" (
    echo ERROR: Icon file assets\icon.ico not found
    goto :error
)

echo Starting build...
echo.

REM Use pyinstaller from virtual environment with spec file
"%VENV_PYINSTALLER%" --noconfirm GaussianHairCube.spec

if %errorlevel% neq 0 (
    echo.
    echo Build failed!
    goto :error
)

echo.
echo ============================================
echo   Build successful!
echo ============================================
echo.
echo Executable location:
echo   dist\GaussianHairCube\GaussianHairCube.exe
echo.
goto :end

:onefile
echo.
echo ============================================
echo   Building GaussianHairCube (Single File Mode)
echo ============================================
echo.

REM Check virtual environment
call :check_venv
if %errorlevel% neq 0 goto :error

REM Check PyInstaller
call :check_pyinstaller
if %errorlevel% neq 0 goto :error

REM Check icon file
if not exist "assets\icon.ico" (
    echo ERROR: Icon file assets\icon.ico not found
    goto :error
)

echo Starting build (single file mode)...
echo.

"%VENV_PYTHON%" build.py --onefile

if %errorlevel% neq 0 (
    echo.
    echo Build failed!
    goto :error
)

echo.
echo ============================================
echo   Build successful!
echo ============================================
echo.
echo Executable location:
echo   dist\GaussianHairCube.exe
echo.
goto :end

:debug
echo.
echo ============================================
echo   Building GaussianHairCube (Debug Mode)
echo ============================================
echo.

REM Check virtual environment
call :check_venv
if %errorlevel% neq 0 goto :error

REM Check PyInstaller
call :check_pyinstaller
if %errorlevel% neq 0 goto :error

REM Check icon file
if not exist "assets\icon.ico" (
    echo ERROR: Icon file assets\icon.ico not found
    goto :error
)

echo Starting build (debug mode)...
echo.

"%VENV_PYTHON%" build.py --debug

if %errorlevel% neq 0 (
    echo.
    echo Build failed!
    goto :error
)

echo.
echo ============================================
echo   Build successful (Debug Mode)!
echo ============================================
echo.
echo Executable location:
echo   dist\GaussianHairCube\GaussianHairCube.exe
echo.
echo NOTE: Debug mode shows console window
echo.
goto :end

:error
echo.
echo Build failed! Please check the error messages above.
echo.
pause
exit /b 1

:end
endlocal