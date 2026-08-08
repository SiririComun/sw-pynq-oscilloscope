import os
import sys
import platform
import subprocess
import urllib.request
from pathlib import Path

class EnvironmentManager:
    """
    Architecture-aware manager for Digilent AD3 WaveForms drivers,
    dependencies, and USB bus permissions on Linux (armhf / aarch64).
    """

    # Driver download URLs by CPU architecture
    DRIVERS = {
        "32bit": {
            "adept_url": "https://files.digilent.com/Software/Adept2%20Runtime/2.27.9/digilent.adept.runtime_2.27.9-armhf.deb",
            "adept_file": "digilent.adept.runtime_2.27.9-armhf.deb",
            "waveforms_url": "https://files.digilent.com/Software/Waveforms/3.25.1/digilent.waveforms_qt5_3.25.1_armhf.deb",
            "waveforms_file": "digilent.waveforms_armhf.deb"
        },
        "64bit": {
            "adept_url": "https://files.digilent.com/Software/Adept2%20Runtime/2.27.9/digilent.adept.runtime_2.27.9-arm64.deb",
            "adept_file": "digilent.adept.runtime_2.27.9-arm64.deb",
            "waveforms_url": "https://files.digilent.com/Software/Waveforms/3.25.1/digilent.waveforms_qt5_3.25.1_arm64.deb",
            "waveforms_file": "digilent.waveforms_arm64.deb"
        }
    }

    @classmethod
    def get_architecture(cls) -> str:
        """Detect CPU architecture: '32bit' for armv7l (Zynq-7000), '64bit' for aarch64 (UltraScale+)."""
        machine = platform.machine().lower()
        if "aarch64" in machine or "arm64" in machine:
            return "64bit"
        return "32bit"

    @classmethod
    def grant_usb_permissions(cls) -> bool:
        """Grant the running user/Jupyter process permissions to access the USB bus."""
        print("[EnvManager] Setting USB bus permissions (/dev/bus/usb)...")
        try:
            subprocess.run(["sudo", "chmod", "-R", "777", "/dev/bus/usb/"], check=True)
            print("[EnvManager] USB permissions successfully granted.")
            return True
        except Exception as e:
            print(f"[EnvManager] Warning: Could not set USB permissions automatically: {e}")
            return False

    @classmethod
    def install_ad3_drivers(cls, download_dir: str = "/tmp/digilent_drivers") -> bool:
        """
        Detects system architecture, downloads Digilent Adept + WaveForms .deb packages,
        installs them via dpkg/apt-get, and sets USB permissions.
        """
        arch = cls.get_architecture()
        driver_info = cls.DRIVERS[arch]
        
        target_dir = Path(download_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        
        adept_path = target_dir / driver_info["adept_file"]
        wf_path = target_dir / driver_info["waveforms_file"]

        print(f"[EnvManager] Detected CPU architecture: '{platform.machine()}' ({arch})")
        print("[EnvManager] Downloading Digilent Adept Runtime and WaveForms packages...")
        
        try:
            if not adept_path.exists():
                urllib.request.urlretrieve(driver_info["adept_url"], adept_path)
            if not wf_path.exists():
                urllib.request.urlretrieve(driver_info["waveforms_url"], wf_path)
                
            print("[EnvManager] Installing Debian packages via dpkg...")
            subprocess.run(["sudo", "dpkg", "-i", str(adept_path)], check=False)
            subprocess.run(["sudo", "dpkg", "-i", str(wf_path)], check=False)
            
            print("[EnvManager] Resolving missing dependencies via apt-get...")
            subprocess.run(["sudo", "apt-get", "install", "-f", "-y"], check=True)
            
            cls.grant_usb_permissions()
            print("\n=== DIGILENT AD3 DRIVER INSTALLATION COMPLETE ===")
            return True
            
        except Exception as e:
            print(f"[EnvManager] Error during driver installation: {e}")
            return False

def install_ad3_drivers():
    """Convenience top-level function."""
    return EnvironmentManager.install_ad3_drivers()

def check_usb_permissions():
    """Convenience top-level function."""
    return EnvironmentManager.grant_usb_permissions()