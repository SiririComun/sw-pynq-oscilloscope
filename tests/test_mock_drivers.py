"""
tests/test_mock_drivers.py: Cloud-compatible unit tests for pynq_oscilloscope.
Validates HardwareTrigger bitmasks, HardwareFilter math, AcousticAnalytics DSP,
audio utilities, and notebook JSON structure without requiring physical FPGA hardware.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from pynq import MMIO
from pynq_oscilloscope.hw_trigger import HardwareTrigger
from pynq_oscilloscope.hw_filter import HardwareFilter
from pynq_oscilloscope.analytics import AcousticAnalytics
from pynq_oscilloscope.fft_dma import StreamingFFT
from pynq_oscilloscope.audio_utils import (
    deinterleave_stereo,
    raw_to_voltages,
    remove_dc_offset,
    normalize_audio,
    audio_to_wav_bytes,
)


# =============================================================================
# 1. Hardware Trigger Register & Bitmask Tests
# =============================================================================

def test_hardware_trigger_bitmasks():
    mmio = MMIO(0x43C10000, 65536)
    trig = HardwareTrigger(mmio)

    # Test Source Selection: Channel 2 (A1) -> sets bit 5
    trig.set_source("CH2")
    assert trig.get_source() == "CH2 (A1)"
    assert bool(mmio.read(0x00) & (1 << 5)) is True

    # Test FFT Channel Routing: Channel 2 (A1) -> sets bit 6
    trig.set_fft_channel("CH2")
    assert trig.get_fft_channel() == "CH2 (A1)"
    assert bool(mmio.read(0x00) & (1 << 6)) is True

    # Test Decimation Factor Configuration (M=50 -> code "11" = 3)
    trig.set_decimation(50)
    assert trig.get_decimation() == 50
    assert (mmio.read(0x14) & 0x3) == 3

    # Test FFT Transform Length Scaling (PG109 format for N=1024 -> NFFT=10)
    trig.set_fft_config(n_points=1024, forward=True)
    assert trig.get_fft_length() == 1024

    # Test Threshold voltage conversion (1.65V -> ~2047 raw -> shifted left by 4)
    trig.set_threshold(1.65)
    assert abs(trig.get_threshold() - 1.65) < 0.01


# =============================================================================
# 2. Hardware Spectral Mask & Frequency Filter Tests
# =============================================================================

def test_hardware_filter_registers():
    mmio = MMIO(0x43C20000, 65536)
    filt = HardwareFilter(mmio, sample_rate_hz=50000.0, fft_points=1024)

    # Test Lowpass / Bass Mode (Cutoff: 250 Hz)
    filt.set_lowpass(cutoff_hz=250.0)
    assert filt.is_enabled is True
    assert filt.mode == "lowpass"
    assert mmio.read(0x04) == 0  # k_start = Bin 0
    assert mmio.read(0x08) == filt.freq_to_bin(250.0)  # k_stop

    # Test Bandpass Mode (300 Hz - 3400 Hz)
    filt.set_bandpass(low_hz=300.0, high_hz=3400.0)
    assert filt.mode == "bandpass"
    assert mmio.read(0x04) == filt.freq_to_bin(300.0)
    assert mmio.read(0x08) == filt.freq_to_bin(3400.0)

    # Test Bypass
    filt.bypass()
    assert filt.is_enabled is False


# =============================================================================
# 3. Acoustic Analytics & DSP Mathematics Tests
# =============================================================================

def test_acoustic_analytics_dsp():
    fs = 50000.0
    analytics = AcousticAnalytics(sample_rate_hz=fs)
    t = np.linspace(0, 0.02, 1000)

    # 1. Test Hilbert Analytic Envelope on a 1 kHz tone modulated at 50 Hz
    carrier = np.sin(2 * np.pi * 1000 * t)
    modulator = 1.0 + 0.5 * np.sin(2 * np.pi * 50 * t)
    am_signal = carrier * modulator
    env = analytics.extract_analytic_envelope(am_signal, remove_dc=True)
    assert len(env) == len(am_signal)
    assert np.mean(env) > 0.5

    # 2. Test Inter-aural Level Difference (ILD)
    sig_left = am_signal * 2.0  # +6 dB louder
    sig_right = am_signal
    ild_db, _, _, _ = analytics.compute_ild(sig_left, sig_right)
    assert np.mean(ild_db) > 5.0

    # 3. Test Sub-Hertz Parabolic Pitch Tracking
    freq_axis = np.fft.rfftfreq(1024, d=1.0 / fs)
    fft_mag = -100.0 * np.ones_like(freq_axis)
    fft_mag[20] = -12.0
    fft_mag[21] = -5.0
    fft_mag[22] = -14.0
    f0, m0 = analytics.track_sub_hertz_pitch(freq_axis, fft_mag, min_freq_hz=100.0)
    assert abs(f0 - freq_axis[21]) < (fs / 1024.0)


# =============================================================================
# 4. Audio Utilities & WAV Encoder Tests
# =============================================================================

def test_audio_utils_pipeline():
    # 1. Test Stereo De-interleaving
    interleaved = np.array([100, 200, 101, 201, 102, 202], dtype=np.uint16)
    ch1, ch2 = deinterleave_stereo(interleaved)
    np.testing.assert_array_equal(ch1, np.array([100, 101, 102]))
    np.testing.assert_array_equal(ch2, np.array([200, 201, 202]))

    # 2. Test Voltage Scaling (12-bit left-aligned to 0-3.3V)
    raw_code = np.array([2047 << 4], dtype=np.uint16)
    voltages = raw_to_voltages(raw_code)
    assert abs(voltages[0] - 1.65) < 0.01

    # 3. Test DC Removal & Normalization
    v = np.array([1.65 + 0.5, 1.65 - 0.5, 1.65 + 0.5, 1.65 - 0.5])
    ac = remove_dc_offset(v)
    assert abs(np.mean(ac)) < 1e-6
    norm = normalize_audio(ac, peak_ref=1.65)
    assert np.max(np.abs(norm)) <= 1.0

    # 4. Test In-Memory WAV Encoding
    audio_mono = np.sin(np.linspace(0, 2 * np.pi * 440, 1000))
    wav_bytes = audio_to_wav_bytes(audio_mono, sample_rate_hz=50000, num_channels=1)
    assert len(wav_bytes) > 44
    assert wav_bytes.startswith(b"RIFF")
    assert b"WAVE" in wav_bytes


# =============================================================================
# 5. Notebook File Integrity Tests
# =============================================================================

def test_notebooks_json_validity():
    repo_root = Path(__file__).resolve().parent.parent
    notebooks = list((repo_root / "notebooks").glob("*.ipynb"))
    assert len(notebooks) >= 5, "Expected at least 5 notebooks in suite"

    for nb_path in notebooks:
        with open(nb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert "cells" in data, f"{nb_path.name} missing 'cells' key"
            assert "nbformat" in data, f"{nb_path.name} missing 'nbformat' key"
            assert len(data["cells"]) > 0, f"{nb_path.name} has no cells"
# =============================================================================
# 6. Hardware Pinning & Release Asset Integrity Gate
# =============================================================================

def test_hardware_json_pinning_integrity():
    """
    Verifies that hardware.json defines a valid repository and version tag,
    and checks that pynq_z2.bit and pynq_z2.hwh exist on GitHub Releases.
    """
    import urllib.request
    import urllib.error

    repo_root = Path(__file__).resolve().parent.parent
    hw_config_path = repo_root / "hardware.json"
    assert hw_config_path.exists(), "hardware.json is missing from repository root!"

    with open(hw_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    repo = config.get("repo")
    version = config.get("version")

    assert repo, "hardware.json must define 'repo'"
    assert version, "hardware.json must define 'version'"
    assert version.startswith("v"), f"Version tag '{version}' must start with 'v'"

    # Verify that both release assets exist and are reachable
    base_url = f"https://github.com/{repo}/releases/download/{version}"
    for asset in ["pynq_z2.bit", "pynq_z2.hwh"]:
        url = f"{base_url}/{asset}"
        req = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=8) as response:
                assert response.status in (200, 302), f"Asset {asset} unreachable at {url}"
        except urllib.error.HTTPError as e:
            assert e.code in (200, 302), f"Asset {url} returned HTTP {e.code}"
        except urllib.error.URLError:
            pytest.skip("Network unreachable, skipping remote release asset verification")