import os
import json
import urllib.request
from pathlib import Path
from pynq import Overlay

class HardwareLoader:
    """
    Smart Overlay Loader that detects the target board and pulls matching
    .bit and .hwh release binaries from GitHub Releases API based on hardware.json.
    """
    
    @staticmethod
    def get_project_root() -> Path:
        """Find the root directory of the package where hardware.json lives."""
        return Path(__file__).resolve().parent.parent

    @classmethod
    def get_hardware_config(cls) -> dict:
        """Load hardware pinning configuration from hardware.json."""
        config_path = cls.get_project_root() / "hardware.json"
        if not config_path.exists():
            return {
                "repo": "SiririComun/hw-xadc-dma-overlays",
                "version": "v1.0.2"
            }
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def get_board_name() -> str:
        """
        Detect active board running PYNQ across all PYNQ OS versions (v2.x, v3.x).
        Normalizes 'PYNQ-Z2' -> 'pynq_z2', 'ZedBoard' -> 'zedboard', etc.
        """
        # 1. Check environment variable set by PYNQ OS ($BOARD)
        board_env = os.environ.get("BOARD")
        if board_env:
            return board_env.lower().replace("-", "_")

        # 2. Check /etc/board.name if on PYNQ Linux
        try:
            if os.path.exists("/etc/board.name"):
                with open("/etc/board.name", "r", encoding="utf-8") as f:
                    return f.read().strip().lower().replace("-", "_")
        except Exception:
            pass

        # 3. Try pynq.Device active device name
        try:
            from pynq import Device
            if Device.active_device and Device.active_device.name:
                return Device.active_device.name.lower().replace("-", "_")
        except Exception:
            pass

        # Fallback default for PYNQ-Z2
        return "pynq_z2"

    @classmethod
    def load_overlay(cls, version: str = None, download_dir: str = None) -> Overlay:
        """
        Detects the host board, fetches matching compiled .bit and .hwh files
        from GitHub Releases if missing, and loads the Overlay into the FPGA fabric.
        """
        config = cls.get_hardware_config()
        repo = config.get("repo", "SiririComun/hw-xadc-dma-overlays")
        target_version = version or config.get("version", "v1.0.2")
        
        board_name = cls.get_board_name()
        bit_filename = f"{board_name}.bit"
        hwh_filename = f"{board_name}.hwh"
        
        # Determine cache location
        if download_dir is None:
            download_dir = Path.home() / ".cache" / "pynq_oscilloscope" / target_version
        else:
            download_dir = Path(download_dir)
            
        download_dir.mkdir(parents=True, exist_ok=True)
        
        local_bit = download_dir / bit_filename
        local_hwh = download_dir / hwh_filename
        
        # Base release URL on GitHub
        base_url = f"https://github.com/{repo}/releases/download/{target_version}/"
        url_bit = f"{base_url}{bit_filename}"
        url_hwh = f"{base_url}{hwh_filename}"
        
        # Download files if they do not exist locally
        if not local_bit.exists() or not local_hwh.exists():
            print(f"[HardwareLoader] Detected target board: '{board_name}'")
            print(f"[HardwareLoader] Downloading overlay '{target_version}' from {repo}...")
            
            try:
                urllib.request.urlretrieve(url_bit, local_bit)
                urllib.request.urlretrieve(url_hwh, local_hwh)
                print("[HardwareLoader] Overlay assets downloaded successfully.")
            except Exception as e:
                print(f"[HardwareLoader] Error downloading release binaries: {e}")
                raise RuntimeError(
                    f"Could not download {bit_filename} from {base_url}. "
                    "Ensure the board has internet access and the hardware release exists."
                ) from e
                
        print(f"[HardwareLoader] Loading overlay into FPGA fabric: {local_bit}")
        return Overlay(str(local_bit))