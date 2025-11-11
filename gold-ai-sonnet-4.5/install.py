#!/usr/bin/env python3
"""
Installation script for Gold AI Sonnet 4.5
Handles MetaTrader5 installation and other dependencies
"""

import subprocess
import sys
import os
import platform

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def install_metatrader5():
    """Install MetaTrader5 package"""
    print("📦 Installing MetaTrader5...")

    # Try different installation methods
    methods = [
        "pip install MetaTrader5",
        "pip install --upgrade MetaTrader5",
        "python -m pip install MetaTrader5",
    ]

    for method in methods:
        if run_command(method, f"Trying: {method}"):
            return True

    # If pip installation fails, try alternative method
    print("⚠️  Standard pip installation failed. Trying alternative method...")

    # Try installing from GitHub or other sources
    alt_methods = [
        "pip install git+https://github.com/metaquotes/MetaTrader5-Python.git",
        "pip install https://github.com/metaquotes/MetaTrader5-Python/archive/master.zip",
    ]

    for method in alt_methods:
        if run_command(method, f"Trying alternative: {method}"):
            return True

    print("❌ MetaTrader5 installation failed. Please install manually:")
    print("   1. Download from: https://www.mql5.com/en/docs/integration/python_metatrader5")
    print("   2. Or try: pip install MetaTrader5")
    print("   3. Make sure MetaTrader 5 terminal is installed on your system")
    return False

def install_requirements():
    """Install Python requirements"""
    print("📦 Installing Python requirements...")

    # First try to install MetaTrader5 separately
    if not install_metatrader5():
        print("⚠️  Continuing with other packages. MetaTrader5 will need manual installation.")

    # Install other requirements
    if os.path.exists('requirements.txt'):
        if run_command("pip install -r requirements.txt", "Installing requirements from requirements.txt"):
            return True
        else:
            print("❌ Failed to install from requirements.txt")
            return False
    else:
        print("❌ requirements.txt not found")
        return False

def check_mt5_terminal():
    """Check if MetaTrader 5 terminal is installed"""
    print("🔍 Checking for MetaTrader 5 terminal...")

    system = platform.system().lower()

    if system == "darwin":  # macOS
        mt5_paths = [
            "/Applications/MetaTrader 5.app",
            "/Applications/MetaTrader 5/MetaTrader 5.app",
            "~/Applications/MetaTrader 5.app"
        ]
    elif system == "windows":
        mt5_paths = [
            "C:\\Program Files\\MetaTrader 5",
            "C:\\Program Files (x86)\\MetaTrader 5"
        ]
    else:  # Linux
        mt5_paths = [
            "/opt/metatrader5",
            "~/metatrader5"
        ]

    found = False
    for path in mt5_paths:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            print(f"✅ MetaTrader 5 found at: {expanded_path}")
            found = True
            break

    if not found:
        print("⚠️  MetaTrader 5 terminal not found in standard locations.")
        print("   Please ensure MetaTrader 5 is installed:")
        if system == "darwin":
            print("   - Download from: https://www.metatrader5.com/en/download")
        elif system == "windows":
            print("   - Download from: https://www.metatrader5.com/en/download")
        else:
            print("   - Check your distribution's package manager")

    return found

def create_env_file():
    """Create .env file from template"""
    if not os.path.exists('.env') and os.path.exists('.env.example'):
        print("📄 Creating .env file from template...")
        try:
            with open('.env.example', 'r') as template:
                with open('.env', 'w') as env_file:
                    env_file.write(template.read())
            print("✅ .env file created. Please edit it with your credentials.")
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
    else:
        print("ℹ️  .env file already exists or .env.example not found")

def main():
    """Main installation process"""
    print("🤖 Gold AI Sonnet 4.5 - Installation")
    print("=" * 50)

    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print(f"❌ Python {python_version.major}.{python_version.minor} is not supported.")
        print("   Please upgrade to Python 3.8 or higher.")
        sys.exit(1)

    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.minor} detected")

    # Check MT5 terminal
    check_mt5_terminal()

    # Install requirements
    if not install_requirements():
        print("❌ Installation failed. Please check the errors above.")
        sys.exit(1)

    # Create environment file
    create_env_file()

    # Create logs directory
    os.makedirs('logs', exist_ok=True)

    print("\n" + "=" * 50)
    print("🎉 Installation completed!")
    print("\n📋 Next steps:")
    print("   1. Edit .env file with your MT5 demo account credentials")
    print("   2. Make sure MetaTrader 5 terminal is running")
    print("   3. Test the system: python test_system.py")
    print("   4. Start trading: python main.py")
    print("   5. Or use web panel: python web_panel.py")
    print("\n📖 For more information, see README.md")

if __name__ == "__main__":
    main()
