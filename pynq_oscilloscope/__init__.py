"""
pynq_oscilloscope: High-level Python library for PYNQ-Z2 Oscilloscope & AD3 integration.
"""

from pynq_oscilloscope.loader import HardwareLoader
from pynq_oscilloscope.env_checker import install_ad3_drivers, check_usb_permissions

__version__ = "1.0.0"
__all__ = ["HardwareLoader", "install_ad3_drivers", "check_usb_permissions"]