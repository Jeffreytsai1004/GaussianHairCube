#!/usr/bin/env python3
"""
Build Script for GaussianHairCube
=================================

Uses the maintained ``GaussianHairCube.spec`` as the single source of truth
for PyInstaller configuration.  This script wraps PyInstaller invocation and
provides convenience commands (clean, installer template).

Usage:
    python build.py                  # build using spec file
    python build.py --clean          # clean build/ and dist/ first
    python build.py --installer      # write installer.iss for Inno Setup
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

SPEC_FILE = "GaussianHairCube.spec"


def clean_build():
    """Remove PyInstaller's build/dist directories and any __pycache__ trees."""
    for d in ("build", "dist"):
        if os.path.exists(d):
            print(f"Removing {d}/")
            shutil.rmtree(d, ignore_errors=True)

    for root, dirs, _ in os.walk("."):
        for d in list(dirs):
            if d == "__pycache__":
                path = os.path.join(root, d)
                print(f"Removing {path}")
                shutil.rmtree(path, ignore_errors=True)


def build_executable(clean: bool = False) -> bool:
    """Invoke PyInstaller with the project's .spec file."""
    if not Path(SPEC_FILE).exists():
        print(f"ERROR: {SPEC_FILE} not found")
        return False

    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm"]
    if clean:
        cmd.append("--clean")
    cmd.append(SPEC_FILE)

    print(f"Running: {' '.join(cmd)}")
    print()
    return subprocess.run(cmd).returncode == 0


def create_installer_script():
    """Emit a starter Inno Setup script for packaging the dist into an installer."""
    script = '''; GaussianHairCube Installer Script — Inno Setup 6.x

#define MyAppName "GaussianHairCube"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "GaussianHairCube"
#define MyAppExeName "GaussianHairCube.exe"

[Setup]
AppId={{REPLACE-WITH-NEW-GUID}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
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
    with open("installer.iss", "w", encoding="utf-8") as f:
        f.write(script)
    print("Wrote installer.iss")


def main():
    parser = argparse.ArgumentParser(description="Build GaussianHairCube")
    parser.add_argument("--clean", action="store_true",
                        help="Clean build/dist before building")
    parser.add_argument("--installer", action="store_true",
                        help="Generate installer.iss template instead of building")
    args = parser.parse_args()

    if args.installer:
        create_installer_script()
        return 0

    if args.clean:
        clean_build()

    ok = build_executable(clean=args.clean)
    if ok:
        print()
        print("=" * 50)
        print("Build complete: dist/GaussianHairCube/GaussianHairCube.exe")
        print("=" * 50)
        return 0
    print("Build failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
