#!/usr/bin/env python3
"""
Build Script for GaussianHairCube
=================================

This script builds the GaussianHairCube application as a Windows executable
using PyInstaller.

Usage:
    python build.py
    
Options:
    --onefile    Create a single executable (slower startup)
    --onedir     Create a directory distribution (default, faster startup)
    --debug      Build with debug console
    --clean      Clean build directories before building
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path


def clean_build():
    """Clean build directories."""
    dirs_to_clean = ['build', 'dist', '__pycache__']
    files_to_clean = ['*.spec']
    
    for d in dirs_to_clean:
        if os.path.exists(d):
            print(f"Removing {d}/")
            shutil.rmtree(d)
    
    for pattern in files_to_clean:
        import glob
        for f in glob.glob(pattern):
            print(f"Removing {f}")
            os.remove(f)
    
    # Clean pycache in subdirectories
    for root, dirs, files in os.walk('.'):
        for d in dirs:
            if d == '__pycache__':
                path = os.path.join(root, d)
                print(f"Removing {path}")
                shutil.rmtree(path)


def build_executable(onefile=False, debug=False):
    """Build the executable using PyInstaller."""
    
    # Base PyInstaller command
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--name=GaussianHairCube',
        '--windowed' if not debug else '--console',
        '--noconfirm',
    ]
    
    # Single file or directory
    if onefile:
        cmd.append('--onefile')
    else:
        cmd.append('--onedir')
    
    # Add icon if exists
    icon_path = Path('assets/icon.ico')
    if icon_path.exists():
        cmd.append(f'--icon={icon_path}')
    
    # Hidden imports
    hidden_imports = [
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'customtkinter',
        'numpy',
        'cv2',
        'scipy',
        'scipy.ndimage',
        'scipy.spatial',
        'sklearn',
        'sklearn.cluster',
        'pygltflib',
    ]
    
    for imp in hidden_imports:
        cmd.append(f'--hidden-import={imp}')
    
    # Collect data
    cmd.append('--collect-data=customtkinter')
    
    # Add data files
    data_files = [
        ('assets', 'assets'),
    ]
    
    for src, dest in data_files:
        if os.path.exists(src):
            cmd.append(f'--add-data={src}{os.pathsep}{dest}')
    
    # Exclude unnecessary modules to reduce size
    excludes = [
        'matplotlib',
        'tkinter.test',
        'unittest',
        'pytest',
        'IPython',
        'jupyter',
        'notebook',
    ]
    
    for exc in excludes:
        cmd.append(f'--exclude-module={exc}')
    
    # Entry point
    cmd.append('main.py')
    
    print("Building GaussianHairCube...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run PyInstaller
    result = subprocess.run(cmd)
    
    return result.returncode == 0


def create_installer_script():
    """Create an Inno Setup script for Windows installer."""
    script = '''
; GaussianHairCube Installer Script
; Requires Inno Setup 6.x

#define MyAppName "GaussianHairCube"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "GaussianHairCube"
#define MyAppURL "https://github.com/your-repo/GaussianHairCube"
#define MyAppExeName "GaussianHairCube.exe"

[Setup]
AppId={{YOUR-GUID-HERE}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
LicenseFile=LICENSE
OutputDir=installer
OutputBaseFilename=GaussianHairCube_Setup_{#MyAppVersion}
SetupIconFile=assets\\icon.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "dist\\GaussianHairCube\\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\\{#MyAppName}"; Filename: "{app}\\{#MyAppExeName}"
Name: "{group}\\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\\{#MyAppName}"; Filename: "{app}\\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
'''
    
    with open('installer.iss', 'w') as f:
        f.write(script)
    
    print("Created installer.iss for Inno Setup")


def main():
    parser = argparse.ArgumentParser(description='Build GaussianHairCube executable')
    parser.add_argument('--onefile', action='store_true', help='Create single executable')
    parser.add_argument('--onedir', action='store_true', help='Create directory distribution (default)')
    parser.add_argument('--debug', action='store_true', help='Build with debug console')
    parser.add_argument('--clean', action='store_true', help='Clean build directories')
    parser.add_argument('--installer', action='store_true', help='Create Inno Setup script')
    
    args = parser.parse_args()
    
    if args.clean:
        clean_build()
        if not any([args.onefile, args.onedir]):
            print("Clean complete.")
            return 0
    
    if args.installer:
        create_installer_script()
        return 0
    
    # Default to onedir if neither specified
    onefile = args.onefile
    
    success = build_executable(onefile=onefile, debug=args.debug)
    
    if success:
        print()
        print("=" * 50)
        print("Build complete!")
        if onefile:
            print("Executable: dist/GaussianHairCube.exe")
        else:
            print("Distribution: dist/GaussianHairCube/")
            print("Run: dist/GaussianHairCube/GaussianHairCube.exe")
        print("=" * 50)
        return 0
    else:
        print("Build failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())