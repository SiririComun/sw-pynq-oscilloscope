"""
pynq_oscilloscope: High-level Python library for PYNQ-Z2 Oscilloscope, Audio Analyzer & AD3 integration.
"""

from pynq_oscilloscope.overlay import OscilloscopeOverlay
from pynq_oscilloscope.hw_trigger import HardwareTrigger
from pynq_oscilloscope.loader import HardwareLoader
from pynq_oscilloscope.env_checker import install_ad3_drivers, check_usb_permissions
from pynq_oscilloscope.xadc_dma import StreamingXADC
from pynq_oscilloscope.fft_dma import StreamingFFT
from pynq_oscilloscope.ad3_wavegen import AD3SignalGenerator
from pynq_oscilloscope.dashboard import OscilloscopeDashboard
from pynq_oscilloscope.audio_dashboard import AudioDashboard
from pynq_oscilloscope.analytics import AcousticAnalytics
from pynq_oscilloscope.notebooks import copy_notebooks
from pynq_oscilloscope.audio_utils import (
    deinterleave_stereo,
    raw_to_voltages,
    remove_dc_offset,
    normalize_audio,
    audio_to_wav_bytes,
    process_raw_recording,
)
from pynq_oscilloscope.hw_filter import HardwareFilter

__version__ = "1.6.0-rc1"
__all__ = [
    "OscilloscopeOverlay",
    "HardwareTrigger",
    "HardwareLoader",
    "install_ad3_drivers",
    "check_usb_permissions",
    "StreamingXADC",
    "StreamingFFT",
    "AD3SignalGenerator",
    "OscilloscopeDashboard",
    "AudioDashboard",
    "AcousticAnalytics",
    "copy_notebooks",
    "HardwareFilter",
]