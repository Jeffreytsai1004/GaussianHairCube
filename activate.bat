@echo off
chcp 65001 >nul 2>nul
REM ============================================
REM GaussianHairCube Virtual Environment Setup
REM ============================================
REM This script creates and activates the virtual environment
REM ============================================

REM Get script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Set virtual environment path
set "VENV_DIR=%SCRIPT_DIR%venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_ACTIVATE=%VENV_DIR%\Scripts\activate.bat"

echo.
echo ============================================
echo   GaussianHairCube Environment Setup
echo ============================================
echo.

REM Check if virtual environment exists
if exist "%VENV_DIR%" (
    echo Virtual environment found: %VENV_DIR%
    echo.
    goto :activate
)

REM Create virtual environment
echo Virtual environment not found, creating...
echo.

REM Check if Python is available
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python not found!
    echo Please install Python and add it to PATH.
    echo.
    pause
    goto :eof
)

REM Show Python version
echo Python version:
python --version
echo.

REM Create venv
echo Creating virtual environment...
python -m venv venv

if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment!
    pause
    goto :eof
)

echo Virtual environment created successfully!
echo.

:activate
REM Check if activate script exists
if not exist "%VENV_ACTIVATE%" (
    echo ERROR: Activation script not found: %VENV_ACTIVATE%
    pause
    goto :eof
)

REM Activate virtual environment
echo Activating virtual environment...
call "%VENV_ACTIVATE%"

REM Check if requirements need to be installed
if exist "%SCRIPT_DIR%requirements.txt" (
    echo.
    echo Checking dependencies...
    
    REM Simple check - try importing customtkinter
    python -c "import customtkinter" >nul 2>nul
    if %errorlevel% neq 0 (
        echo.
        echo Installing dependencies from requirements.txt...
        echo This may take a few minutes...
        echo.
        pip install -r "%SCRIPT_DIR%requirements.txt"
        
        if %errorlevel% neq 0 (
            echo.
            echo WARNING: Some dependencies may have failed to install.
            echo Please check the error messages above.
            echo.
        ) else (
            echo.
            echo Dependencies installed successfully!
        )
    ) else (
        echo Dependencies already installed.
    )
)

echo.
echo ============================================
echo   Environment Ready!
echo ============================================
echo.
echo You can now:
echo   - Run the app:    python main.py
echo   - Build the app:  build.bat
echo   - Deactivate:     deactivate
echo.

REM Open a new command prompt with the activated environment
cmd /k