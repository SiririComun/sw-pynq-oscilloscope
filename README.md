# Real-Time Dual-Channel Oscilloscope & Audio Spectrum Analyzer

[![PyPI Version](https://img.shields.io/pypi/v/pynq-oscilloscope.svg)](https://pypi.org/project/pynq-oscilloscope/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/SiririComun/sw-pynq-oscilloscope/blob/main/LICENSE)
[![Hardware Overlay](https://img.shields.io/badge/Hardware-hw--xadc--dma--overlays%20v1.4.0-orange.svg)](https://github.com/SiririComun/hw-xadc-dma-overlays)
[![Board Support](https://img.shields.io/badge/Board-PYNQ--Z2-green.svg)](https://tul.com.tw/ProductsPYNQ-Z2.html)

A high-performance, dark-mode real-time **Dual-Channel Oscilloscope and Audio Spectrum Analyzer** software stack running natively on PYNQ Linux platforms. 

Features **simultaneous dual-channel continuous analog acquisition** on Arduino header pins **A0** (`Vaux1`) and **A1** (`Vaux9`), **FPGA-accelerated anti-aliasing audio decimation ($50\,\text{kSPS}$ audio rate / $40.96\,\text{ms}$ window)**, **sub-sample trigger phase-locking**, **dedicated passive microphone instruments (`AudioDashboard`)**, **high-resolution audio FFT ($\Delta f \approx 24.41\,\text{Hz}$)** with sub-Hertz quadratic pitch tracking, and non-blocking dual-channel analog waveform generation with the **Digilent Analog Discovery 3 (AD3)** via `pydwf`.

---

## 🏛 System Architecture

This repository adopts the **canonical PYNQ Custom Overlay pattern** (`OscilloscopeOverlay`). It automatically pulls its compiled hardware bitstream and metadata from GitHub Releases (or loads local custom `.bit` builds) and encapsulates the Dual DMA receivers, AXI-Lite trigger registers, sequencer controls, and dual-wavegen into a unified Python object.

```
 [ Analog Discovery 3 ] ──(W1: Yellow)──────> [ PYNQ-Z2 Pin A0 (Vaux1) ]
 [      Wavegen       ] ──(W2: Yellow/White)─> [ PYNQ-Z2 Pin A1 (Vaux9) ]
 [         OR         ]                                       │
 [ MAX4466 Mics A0/A1 ]                       (XADC Dual Continuous Sequencer)
        │                                                     │ (1 MSPS Interleaved Stream)
  (pydwf SDK)                                                 ▼
        │                                            [ axis_trigger_unit IP ]
        ▼                                            (Selectable Trigger Source: A0/A1)
 [ AD3SignalGenerator ]                                       │ (Gated Stream)
 (Concurrent W1 & W2)                                         ▼
                                                     [ axis_decimator IP ]
                                                  (M = 10x Anti-Aliasing Averaging)
                                                              │ (50 kSPS Audio Stream)
                                                              ▼
                                                     [ tlast_generator (2048 pts / 40.96 ms) ]
                                                              │
                                                     [ axis_broadcaster ]
                                               ┌──────────────┴──────────────┐
                                               ▼ (40.96 ms Time Stream)      ▼ (Signed Stream w/ DC Block)
                                      [ AXI DMA 0 (Time) ]          [ xfft (2048-pt BFP) ]
                                               │                             │ (Δf = 24.41 Hz)
                                               │                    [ CORDIC (Magnitude) ]
                                               │                             │
                                               │                    [ AXI DMA 1 (FFT) ]
                                               │                             │
                                               └──────────────┬──────────────┘
                                                              ▼
                                                   [ OscilloscopeOverlay ]
                                            ├── .trigger          (HardwareTrigger AXI-Lite)
                                            ├── .xadc             (StreamingXADC DMA Driver)
                                            ├── .fft              (StreamingFFT PL DMA Driver)
                                            ├── .wavegen          (AD3SignalGenerator Dual-DAC)
                                            ├── .dashboard()      (AD3 Laboratory Scope UI)
                                            └── .audio_dashboard()(Dedicated Passive Microphone UI)
```

---

## 🖥 Interactive Dashboards

### 1. Dedicated Microphone & Audio Instrument (`ol.audio_dashboard()`)
Designed specifically for passive **MAX4466 electret microphones** (or any analog audio sensor on pins **A0** and **A1**), running completely independently without requiring an AD3:
* **$40.96\,\text{ms}$ Audio Timebase:** Displays multi-cycle acoustic waveforms for speech, musical instruments, and bass frequencies ($50\,\text{Hz} - 250\,\text{Hz}$).
* **Live VU Meters & Clipping Alerts:** Status bar indicators that flash red if either microphone saturates ($V < 0.10\,\text{V}$ or $V > 3.10\,\text{V}$).
* **Sub-Bin Quadratic Peak Pitch Tracking:** Extracts the dominant acoustic fundamental ($f_0$) with $\pm 0.5\,\text{Hz}$ accuracy.

### 2. Laboratory Oscilloscope Instrument (`ol.dashboard()`)
Runs concurrent non-blocking dual waveform generation with the Analog Discovery 3 (W1 & W2) and streaming oscilloscope analysis with live 5-period auto-ranging.

---

## 🔌 Hardware Setup & Wiring

### Option A: Dual MAX4466 Microphones (Acoustic Audio Mode)
| MAX4466 Pin | PYNQ-Z2 Connection | Description |
| :--- | :--- | :--- |
| **`VCC`** | **`3.3V`** (Power Header) | Supply rail ($2.4\,\text{V} - 5.5\,\text{V}$) |
| **`GND`** | **`GND`** (Power Header) | Common analog ground |
| **`OUT` (Mic 1)** | **Header `J1` Pin A0** | Channel 1 Audio Input ($1.65\,\text{V}$ resting bias) |
| **`OUT` (Mic 2)** | **Header `J1` Pin A1** | Channel 2 Audio Input ($1.65\,\text{V}$ resting bias) |

### Option B: Analog Discovery 3 (Signal Generator Mode)
| AD3 Wire | Wire Color | PYNQ-Z2 Analog Pin | Signal Description |
| :--- | :--- | :--- | :--- |
| **Wavegen 1 (W1)** | Solid Yellow | **Header `J1` Pin A0** | Channel 1 Analog Input (`Vaux1`) |
| **Wavegen 2 (W2)** | Yellow / White Stripe | **Header `J1` Pin A1** | Channel 2 Analog Input (`Vaux9`) |
| **GND** | Solid Black | **PYNQ-Z2 GND** | Common Analog Reference |

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

---

## 💻 Python API Usage

### Launch Interactive Microphone Audio Dashboard
```python
from pynq_oscilloscope import check_usb_permissions, OscilloscopeOverlay

check_usb_permissions()

# Automatically loads v1.4.0 overlay and programs FPGA
ol = OscilloscopeOverlay()

# Launch dedicated dark-mode Audio & Microphone Instrument
app = ol.audio_dashboard()
```

### Programmatic Dual-Channel Capture
```python
from pynq_oscilloscope import OscilloscopeOverlay

ol = OscilloscopeOverlay()

# Configure Trigger: Trigger on Channel 1 (A0) Rising Edge @ 1.65V
ol.trigger.configure(mode="Auto", edge="Rising", source="CH1", threshold_volts=1.65)

# Synchronously capture both channels (1024 samples per channel @ 50 kSPS, 40.96 ms window)
v_a0, v_a1 = ol.capture_stereo()
print(f"Captured A0 Vpp: {v_a0.max()-v_a0.min():.2f} V | A1 Vpp: {v_a1.max()-v_a1.min():.2f} V")

ol.close()
```

---

## 📓 Notebook Suite

| Notebook | Description | Key Modules Used |
| :--- | :--- | :--- |
| **`01_ad3_getting_started.ipynb`** | Verifies Digilent drivers and generates analog waveforms in background worker. | `AD3SignalGenerator`, `check_usb_permissions` |
| **`02_xadc_getting_started.ipynb`** | Single-channel hardware triggering and DMA capture on A0. | `OscilloscopeOverlay`, `HardwareTrigger` |
| **`03_oscilloscope_dashboard.ipynb`** | **Laboratory Instrument:** Deploys interactive Dual-Channel AD3 Oscilloscope Dashboard. | `OscilloscopeOverlay` |
| **`04_fft_spectrum_analyzer.ipynb`** | **Spectrum Analyzer Guide:** Captures PL hardware FFT spectra, analyzes harmonics. | `OscilloscopeOverlay`, `StreamingFFT` |
| **`05_audio_dashboard.ipynb`** | **Audio Instrument:** Deploys dedicated Audio & Microphone Dashboard (`ol.audio_dashboard()`). | `OscilloscopeOverlay`, `AudioDashboard` |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/SiririComun/sw-pynq-oscilloscope/blob/main/LICENSE) file for details.