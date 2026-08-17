# Real-Time Simultaneous Dual-Channel Oscilloscope & Spectrum Analyzer

[![PyPI Version](https://img.shields.io/pypi/v/pynq-oscilloscope.svg)](https://pypi.org/project/pynq-oscilloscope/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/SiririComun/sw-pynq-oscilloscope/blob/main/LICENSE)
[![Hardware Overlay](https://img.shields.io/badge/Hardware-hw--xadc--dma--overlays%20v1.3.0--rc1-orange.svg)](https://github.com/SiririComun/hw-xadc-dma-overlays)
[![Board Support](https://img.shields.io/badge/Board-PYNQ--Z2-green.svg)](https://tul.com.tw/ProductsPYNQ-Z2.html)

A high-performance, dark-mode real-time **Dual-Channel Oscilloscope and Spectrum Analyzer** software stack running natively on PYNQ Linux platforms. 

Features **simultaneous dual-channel continuous analog acquisition** on Arduino header pins **A0** (`Vaux1`) and **A1** (`Vaux9`), **selectable hardware edge triggering (`CH1 / A0` vs `CH2 / A1`)**, **FPGA-accelerated 2048-point FFT & CORDIC magnitude extraction** in Programmable Logic, **dual AXI DMA streaming direct to DDR memory at ~30 FPS**, and non-blocking concurrent dual-channel analog waveform generation with the **Digilent Analog Discovery 3 (AD3)** via `pydwf`.

---

## 🏛 System Architecture

This repository adopts the **canonical PYNQ Custom Overlay pattern** (`OscilloscopeOverlay`). It automatically pulls its compiled hardware bitstream and metadata from GitHub Releases (or loads local custom `.bit` builds) and encapsulates the Dual DMA receivers, AXI-Lite trigger registers, sequencer controls, and dual-wavegen into a unified Python object.

```
 [ Analog Discovery 3 ] ──(W1: Yellow)──────> [ PYNQ-Z2 Pin A0 (Vaux1) ]
 [      Wavegen       ] ──(W2: Yellow/White)─> [ PYNQ-Z2 Pin A1 (Vaux9) ]
        │                                                     │
  (pydwf SDK)                                     (XADC Dual Continuous Sequencer)
        │                                                     │ (1 MSPS Interleaved Stream)
        ▼                                                     ▼
 [ AD3SignalGenerator ]                              [ axis_trigger_unit IP ]
 (Concurrent W1 & W2)                                (Selectable Trigger Source: A0/A1)
                                                              │
                                                     [ tlast_generator (2048 pts) ]
                                                              │
                                                     [ axis_broadcaster ]
                                               ┌──────────────┴──────────────┐
                                               ▼ (Time Stream)               ▼ (Signed Stream)
                                      [ AXI DMA 0 (Time) ]          [ xfft (2048-pt BFP) ]
                                               │                             │
                                               │                    [ CORDIC (Magnitude) ]
                                               │                             │
                                               │                    [ AXI DMA 1 (FFT) ]
                                               │                             │
                                               └──────────────┬──────────────┘
                                                              ▼
                                                   [ OscilloscopeOverlay ]
                                            ├── .trigger  (HardwareTrigger AXI-Lite)
                                            ├── .xadc     (StreamingXADC DMA Driver)
                                            ├── .fft      (StreamingFFT PL DMA Driver)
                                            ├── .wavegen  (AD3SignalGenerator Dual-DAC)
                                            └── .dashboard() (Interactive 4-Tab Instrument)
```

---

## 🖥 Interactive 4-Tab Dashboard UI Guide

### Tab 1 — 📈 Dual Oscilloscope (A0 & A1)
Displays synchronized real-time time-domain traces for **Channel 1 (A0, Cyan)** and **Channel 2 (A1, Magenta)**. The **Trigger Threshold Line (Orange Dashed)** automatically relocates to whichever channel is selected as the active trigger source.

![Dual Scope Triggered on A0](https://raw.githubusercontent.com/SiririComun/sw-pynq-oscilloscope/main/docs/images/dashboard_dual_scope_a0.png)

*Trigger source set to **CH2 (A1)** — the trigger threshold line dynamically moves to the bottom A1 subplot:*
![Dual Scope Triggered on A1](https://raw.githubusercontent.com/SiririComun/sw-pynq-oscilloscope/main/docs/images/dashboard_dual_scope_a1.png)

---

### Tab 2 — 📊 Dual Spectrum Analyzer (FFTs of A0 & A1)
Computes and renders high-speed frequency spectra for both channels simultaneously ($0 - 250\,\text{kHz}$) with automated fundamental peak frequency ($f_0$) tracking.

![Dual FFT Spectrum Analyzer](https://raw.githubusercontent.com/SiririComun/sw-pynq-oscilloscope/main/docs/images/dashboard_dual_fft.png)

---

### Tab 3 & 4 — 🔀 Dedicated Channel Views
Stacked multi-domain displays showing Time Domain (top) and Frequency Domain (bottom) for each channel individually:

| Channel 1: A0 (Time + Spectrum) | Channel 2: A1 (Time + Spectrum) |
| :---: | :---: |
| ![Channel 1 View](https://raw.githubusercontent.com/SiririComun/sw-pynq-oscilloscope/main/docs/images/dashboard_ch1_view.png) | ![Channel 2 View](https://raw.githubusercontent.com/SiririComun/sw-pynq-oscilloscope/main/docs/images/dashboard_ch2_view.png) |

---

## ⚙️ Control Panel Reference

1. **Action Row (Row 1):**
   * **`▶ Start Live`:** Launches non-blocking dual DMA stream and AD3 dual wavegen.
   * **`■ Stop`:** Cleanly halts hardware loops, disarms triggers, and releases AD3 handles.
   * **`⚡ Force / Arm`:** Forces capture in Auto/Normal mode or arms single-shot capture.
   * **`Auto-Range` (Toggle):** Dynamically scales horizontal timebase (5–10 signal periods) according to the selected trigger channel's frequency.
   * **`Live Metric Bar`:** Real-time $V_{pp}$ and peak fundamental frequency ($f_0$) for both A0 and A1.

2. **Trigger Controls (Row 2):**
   * **`Trig Mode`:** `Auto` (continuous with 50 ms fallback), `Normal` (strictly edge-gated), `Single` (transient capture).
   * **`Trig Edge`:** `Rising` / `Falling` edge slope detection.
   * **`Trig Source`:** Selects active hardware trigger channel: **`CH1 (A0)`** or **`CH2 (A1)`**.
   * **`Trig Level`:** Sets the analog threshold register ($0.0\,\text{V} - 3.3\,\text{V}$).

3. **Dual Wavegen Controls (Rows 3 & 4):**
   * **`CH1 (A0)` & `CH2 (A1)`:** Independent waveform shapes (`Sine`, `Triangle`, `Square`), amplitudes ($0.1\,\text{V} - 1.5\,\text{V}$), and frequencies ($50\,\text{Hz} - 100\,\text{kHz}$).

4. **FFT Controls (Row 5):**
   * **`FFT Unit`:** Select logarithmic power (`dBV`, `dBFS`) or linear amplitude (`Linear`).
   * **`Span / Zoom`:** Zooms frequency axis (`Full 250 kHz`, `100 kHz`, `20 kHz`).

---

## 🔌 Hardware Setup & Wiring

| AD3 Wire | Wire Color | PYNQ-Z2 Analog Pin | Signal Description |
| :--- | :--- | :--- | :--- |
| **Wavegen 1 (W1)** | Solid Yellow | **Header `J1` Pin A0** (Pin 6 - Bottom) | Channel 1 Analog Input (`Vaux1`) |
| **Wavegen 2 (W2)** | Yellow / White Stripe | **Header `J1` Pin A1** (Pin 5 - 2nd from Bottom) | Channel 2 Analog Input (`Vaux9`) |
| **GND** | Solid Black | **PYNQ-Z2 GND** | Common Analog Reference |

*Note: Connect the AD3 USB cable to the large rectangular **USB HOST** port on the PYNQ-Z2 board. Use an external 5V auxiliary power supply for the AD3 to ensure voltage rail stability under dual-channel generation.*

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

### Launch Interactive Dual Dashboard
```python
from pynq_oscilloscope import check_usb_permissions, OscilloscopeOverlay

check_usb_permissions()

# Automatically loads v1.3.0-rc1 dual-channel overlay and programs FPGA
ol = OscilloscopeOverlay()

# Launch interactive 4-tab Plotly + IPywidgets dashboard
app = ol.dashboard()
```

### Programmatic Dual-Channel Capture
```python
from pynq_oscilloscope import OscilloscopeOverlay

ol = OscilloscopeOverlay()

# Configure Trigger: Trigger on Channel 2 (A1) Rising Edge @ 1.65V
ol.trigger.configure(mode="Auto", edge="Rising", source="CH2", threshold_volts=1.65)

# Synchronously capture both channels (1024 samples per channel @ 500 kSPS)
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
| **`03_oscilloscope_dashboard.ipynb`** | **Main Application:** Deploys the complete interactive 4-Tab Dual-Channel Dashboard. | `OscilloscopeOverlay` |
| **`04_fft_spectrum_analyzer.ipynb`** | **Spectrum Analyzer Guide:** Captures PL hardware FFT spectra, analyzes harmonics. | `OscilloscopeOverlay`, `StreamingFFT` |

---

## ⚠️ Release Notes & Disclaimer

> **Pre-Release Disclaimer (`v1.3.0-rc1`):**
> 
> * **Sampling Rate:** The dual-channel continuous sequencer operates at an aggregate sampling rate of $1.0\,\text{MSPS}$ ($500\,\text{kSPS}$ per channel).
> * **Hardware Phase-Lock:** The hardware trigger unit guarantees that DMA packets strictly start aligned to Channel 1 (A0) even when triggering off Channel 2 (A1).
> * **Known Considerations:** While fully validated for standard laboratory wave generation and cross-channel triggering up to $100\,\text{kHz}$, fine adjustments for extreme frequency ratios (e.g. $>10:1$ differences between A0 and A1) or high-noise edge environments may be further tuned in future revisions.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/SiririComun/sw-pynq-oscilloscope/blob/main/LICENSE) file for details.