# Real-Time 1 MSPS Hardware-Triggered Oscilloscope & Spectrum Analyzer

[![PyPI Version](https://img.shields.io/pypi/v/pynq-oscilloscope.svg)](https://pypi.org/project/pynq-oscilloscope/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hardware Overlay](https://img.shields.io/badge/Hardware-hw--xadc--dma--overlays%20v1.2.0-orange.svg)](https://github.com/SiririComun/hw-xadc-dma-overlays)
[![Board Support](https://img.shields.io/badge/Board-PYNQ--Z2-green.svg)](https://tul.com.tw/ProductsPYNQ-Z2.html)

A high-performance, dark-mode real-time Oscilloscope and Spectrum Analyzer software stack running natively on PYNQ Linux platforms. 

Features **sub-microsecond hardware edge triggering** (`axis_trigger_unit`), **FPGA-accelerated 2048-point FFT & CORDIC magnitude extraction** in Programmable Logic, **dual AXI DMA streaming direct to DDR memory at ~290 FPS**, and non-blocking analog signal generation with the **Digilent Analog Discovery 3** via `pydwf`.

---

## 🏛 System Architecture

This repository adopts the **canonical PYNQ Custom Overlay pattern** (`OscilloscopeOverlay`). It automatically pulls its compiled hardware bitstream and metadata from GitHub Releases (or loads local custom `.bit` builds) and encapsulates the Dual DMA receivers, AXI-Lite trigger registers, and wavegen into a unified Python object.

```
 [ Analog Discovery 3 (W1) ] ──(Analog Jumper Wire)──> [ PYNQ-Z2 Header (A0) ]
              │                                                     │
        (pydwf SDK)                                     (XADC 1 MSPS AXI-Stream)
              │                                                     │
              ▼                                                     ▼
    [ AD3SignalGenerator ]                               [ axis_trigger_unit IP ]
              │                                                     │
              │                                          [ tlast_generator (2048 pts) ]
              │                                                     │
              │                                          [ axis_broadcaster ]
              │                                    ┌────────────────┴────────────────┐
              │                                    ▼ (Time Stream)                   ▼ (Signed Stream)
              │                           [ AXI DMA 0 (Time) ]              [ xfft (2048-pt BFP) ]
              │                                    │                                 │
              │                                    │                        [ CORDIC (Magnitude) ]
              │                                    │                                 │
              │                                    │                        [ AXI DMA 1 (FFT) ]
              │                                    │                                 │
              └────────────────────────────────────┴────────────────┬────────────────┘
                                                                    ▼
                                                         [ OscilloscopeOverlay ]
                                                  ├── .trigger  (HardwareTrigger AXI-Lite)
                                                  ├── .xadc     (StreamingXADC DMA Driver)
                                                  ├── .fft      (StreamingFFT PL DMA Driver)
                                                  ├── .wavegen  (AD3SignalGenerator)
                                                  └── .dashboard() (Interactive Multi-Tab Instrument)
```

---

## 🖥 Interactive Dashboard UI Guide

![Real-Time Oscilloscope & Spectrum Analyzer Dashboard](https://raw.githubusercontent.com/SiririComun/sw-pynq-oscilloscope/main/docs/images/dashboard_screenshot.png)

### 1. Action & Status Bar (Row 1)
* **`▶ Start`:** Initializes background acquisition, arms the FPGA trigger, and streams Time & Frequency domains concurrently at up to 30 FPS.
* **`■ Stop`:** Cleanly halts the acquisition loop, disarms the trigger, stops the AD3 wavegen, and releases device handles.
* **`⚡ Force / Arm`:** 
  * In **Single Mode**, re-arms the trigger to capture the next single transient event.
  * In **Auto/Normal Mode**, forces an immediate hardware frame capture.
* **`🗑 Clear Log`:** Instantly clears the console output area below the dashboard.
* **`Auto-Range` (Toggle):** Dynamically scales the horizontal timebase (5–10 signal periods) and adapts the vertical Y-axis limits.
* **`Live Vpp` & `Peak f0`:** Real-time peak-to-peak voltage calculation and automated fundamental frequency tracking.

### 2. Hardware Trigger Controls (Row 2)
* **`Trig Mode`:**
  * **`Auto`:** Continuous live stream. Locks onto trigger edges; if no edge occurs within 50 ms (disconnected input or threshold out of range), the 50 ms hardware timeout forces a capture so the display never freezes.
  * **`Normal`:** Strictly edge-triggered. Freezes and holds the last frame when no trigger edge is present.
  * **`Single`:** Captures **one single frame** on the first trigger event and freezes. Re-arm by clicking **`⚡ Force / Arm`**.
* **`Trig Edge` (`Rising` / `Falling`):** Configures the FPGA voltage comparator slope.
* **`Trig Level` (Slider & Numeric Box):** Sets the FPGA threshold register (`0x08`) between $0.0\,\text{V}$ and $3.3\,\text{V}$ with client-side zero-latency linking (`widgets.jslink`).

### 3. AD3 Signal Generator & FFT Controls (Rows 3, 4 & 5)
* **`Waveform` (`Sine`, `Triangle`, `Square`):** Selects DAC output waveform on AD3 W1.
* **`Amp` & `Freq` Sliders:** Adjusts output amplitude ($0.1\,\text{V} - 1.5\,\text{V}$) and frequency ($100\,\text{Hz} - 250\,\text{kHz}$) on the fly.
* **`FFT Unit` (`dBV`, `dBFS`, `Linear`):** Selects logarithmic power or linear amplitude for the spectrum analyzer.
* **`Span / Zoom` (`Full 500 kHz`, `100 kHz`, `20 kHz`):** Zooms the frequency horizontal axis.

### 4. Multi-Tab Instrument Display
* **📈 Tab 1 — Oscilloscope:** 1 MSPS time-domain trace + live orange dashed trigger threshold line.
* **📊 Tab 2 — Spectrum Analyzer:** Real-time PL FFT spectrum with cyan diamond fundamental frequency marker ($f_0$).
* **🔀 Tab 3 — Dual View:** Synchronized stacked display showing Time Domain (top) and Frequency Domain (bottom) simultaneously.

---

## 🔌 Hardware Setup & Wiring

1. **AD3 USB Connection:**
   * Plug the Analog Discovery 3 USB cable into the large rectangular **USB HOST** port on the PYNQ-Z2 board (adjacent to Ethernet).
2. **USB Cable Quality:**
   * Ensure you use a **Data + Power USB-C cable** (charging-only cables will not be detected by Linux).
3. **Power Supply:**
   * Power the AD3 with an external **5V auxiliary power supply** to prevent brownouts under load.
4. **Analog Signals:**
   * Connect a jumper wire from **Wavegen 1 (W1)** on the AD3 to **Analog Input A0** on the PYNQ-Z2 Arduino header.
   * Connect an AD3 **GND** pin to a PYNQ-Z2 **GND** pin.

---

## 🚀 Quick Start & Installation

### 1. Install Package from PyPI
```bash
pip install --upgrade pynq-oscilloscope
```

### 2. Copy Example Notebooks to Jupyter Workspace
```bash
pynq-oscilloscope-get-notebooks
```

### 3. Install Digilent AD3 Drivers (One-Time Setup)
```python
from pynq_oscilloscope import install_ad3_drivers
install_ad3_drivers()
```

---

## 💻 Python API Usage

### 1. Launch Interactive Dashboard in 2 Lines
```python
from pynq_oscilloscope import OscilloscopeOverlay

# Automatically downloads v1.2.0 release and programs FPGA
ol = OscilloscopeOverlay()

# Launch dark-mode interactive Plotly + IPywidgets instrument
app = ol.dashboard()
```

### 2. Programmatic Time & Frequency Domain DMA Capture
```python
from pynq_oscilloscope import OscilloscopeOverlay
from pynq_oscilloscope.fft_dma import StreamingFFT

ol = OscilloscopeOverlay()

# Configure FPGA Trigger: Rising Edge @ 1.65V
ol.trigger.configure(mode="Auto", edge="Rising", threshold_volts=1.65)

# Synchronous capture of both Time and Frequency domains (0 ms dead time)
voltages, freqs, mags = ol.capture_both(unit="dBV")

peak_f, peak_m = StreamingFFT.get_peak_frequency(freqs, mags)
print(f"Captured {len(voltages)} time samples. Vpp: {voltages.max()-voltages.min():.2f}V")
print(f"Dominant Peak: {peak_f/1e3:.2f} kHz @ {peak_m:.1f} dBV")

ol.close()
```

---

## 📓 Notebook Suite

| Notebook | Description | Key Modules Used |
| :--- | :--- | :--- |
| **`01_ad3_getting_started.ipynb`** | Verifies Digilent drivers and generates analog waveforms in background worker. | `AD3SignalGenerator`, `check_usb_permissions` |
| **`02_xadc_getting_started.ipynb`** | Demonstrates `OscilloscopeOverlay`, hardware triggering, and time-domain DMA capture. | `OscilloscopeOverlay`, `HardwareTrigger` |
| **`03_oscilloscope_dashboard.ipynb`** | **Main Application:** Deploys the complete interactive Multi-Tab Oscilloscope & Spectrum Analyzer Dashboard. | `OscilloscopeOverlay` |
| **`04_fft_spectrum_analyzer.ipynb`** | **Spectrum Analyzer Guide:** Captures PL hardware FFT spectra, analyzes harmonics (Sine vs. Square). | `OscilloscopeOverlay`, `StreamingFFT` |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.