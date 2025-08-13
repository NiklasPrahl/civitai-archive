#!/usr/bin/env python3
"""
Installation script for Civitai Manager web dependencies
"""

import subprocess
import sys


def install_package(package):
    """Install a Python package with pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ {package} successfully installed")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Error installing {package}")
        return False

def main():
    print("=" * 60)
    print("Civitai Manager Web Dependencies Installation")
    print("=" * 60)
    
    # List of required packages
    packages = [
        "flask>=3.0.0",
        "flask-wtf>=1.2.0", 
        "wtforms>=3.1.0",
        "werkzeug>=3.0.0"
    ]
    
    print("Installing required web packages...")
    print()
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
        print()
    
    print("=" * 60)
    if success_count == len(packages):
        print("✓ All packages successfully installed!")
        print()
        print("You can now start the web application:")
        print("  python start_web.py")
        print("  or")
        print("  python -m civitai_manager.main --web")
    else:
        print(f"⚠ {len(packages) - success_count} packages could not be installed.")
        print("Try manual installation:")
        print("  pip install -r requirements.txt")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 