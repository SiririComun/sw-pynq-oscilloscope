# Real-Time Dual-Channel Multi-Regime Oscilloscope & Audio Spectrum Analyzer

[![PyPI Version](https://img.shields.io/pypi/v/pynq-oscilloscope.svg)](https://pypi.org/project/pynq-oscilloscope/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/SiririComun/sw-pynq-oscilloscope/blob/main/LICENSE)
[![Hardware Overlay](https://img.shields.io/badge/Hardware-hw--xadc--dma--overlays%20v1.4.5-orange.svg)](https://github.com/SiririComun/hw-xadc-dma-overlays)
[![Board Support](https://img.shields.io/badge/Board-PYNQ--Z2-green.svg)](https://tul.com.tw/ProductsPYNQ-Z2.html)

A high-performance, dark-mode real-time **Dual-Channel Multi-Regime Oscilloscope and Audio Spectrum Analyzer** software stack running natively on PYNQ Linux platforms. 

Features **runtime-selectable operating regimes** (Wideband Lab Scope, Full Audio, Speech, Deep Bass Zoom), **FPGA-accelerated anti-aliasing decimation ($M \in \{1, 10, 20, 50\}$)**, **runtime $N$-point FFT transform length scaling ($N \in \{512, 1024, 2048\}$)**, **sub-sample trigger phase-locking**, **dedicated passive microphone instruments (`AudioDashboard`)**, multi-windowing (Hann, Hamming, Blackman, Flat-Top), sub-Hertz quadratic pitch tracking, and direct Jupyter audio playback (`ol.play_audio()`).

---

## 🏛 System Architecture

This repository adopts the **canonical PYNQ Custom Overlay pattern** (`OscilloscopeOverlay`). It automatically pulls its compiled hardware bitstream and metadata from GitHub Releases (or loads local custom `.bit` builds) and encapsulates the Dual DMA receivers, AXI-Lite trigger registers, sequencer controls, dynamic decimators, and dual-wavegen into a unified Python object.

```
 [ Analog Discovery 3 ] ──(W1: Yellow)──────> [ PYNQ-Z2 Pin A0 (Vaux1) ]
 [      Wavegen       ] ──(W2: Yellow/White)─> [ PYNQ-Z2 Pin A1 (Vaux9) ]
 [         OR         ]                                       │
 [ MAX4466 Mics A0/A1 ]                       (XADC Dual Continuous Sequencer)
        │                                                     │ (1 MSPS Interleaved Stream)
  (pydwf SDK)                                                 ▼
        │                                            [ axis_trigger_unit IP ]
        ▼                                      (Selectable Trigger Source: A0 / A1)
 [ AD3SignalGenerator ]                                       │ (Gated Stream)
 (Concurrent W1 & W2)                                         ▼
                                                     [ axis_decimator IP ]
                                           (Programmable M = 1, 10, 20, 50 in PL)
                                                              │ (Audio / Decimated Stream)
                                                              ▼
                                                     [ tlast_generator (Programmable N) ]
                                                              │ (w/ TLAST)
                                                     [ axis_broadcaster ]
                                               ┌──────────────┴──────────────┐
                                               ▼ (Decimated Time Stream)     ▼ (Signed Stream w/ DC Block)
                                      [ AXI DMA 0 (Time) ]          [ xfft (Runtime N Core) ]
                                               │                             │ (N = 512, 1024, 2048)
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
                                            ├── .set_profile()    (Multi-Regime Runtime Switcher)
                                            ├── .play_audio()     (Jupyter Audio Playback)
                                            ├── .dashboard()      (AD3 Laboratory Scope UI)
                                            └── .audio_dashboard()(Dedicated Passive Microphone UI)
```

---

## 🎛 Multi-Regime Operating Profiles

The system seamlessly reconfigures sampling rate, packet duration, and FFT resolution on the fly via `ol.set_profile()`:

| Profile Name | Decimator ($M$) | Transform ($N$) | Sampling Rate ($f_s$) | Nyquist Bandwidth | Time Window ($T_{\text{win}}$) | Resolution ($\Delta f$) | Best Used For |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`oscilloscope`** | **$1$** (Bypass) | $2048$ | $500\,\text{kSPS}$ | $0 - 250\,\text{kHz}$ | $2.05\,\text{ms}$ | $244.14\,\text{Hz}$ | Function generators, high-speed pulses, logic edges |
| **`audio`** | **$10$** | $2048$ | $50\,\text{kSPS}$ | $0 - 25\,\text{kHz}$ | $40.96\,\text{ms}$ | $24.41\,\text{Hz}$ | Full-spectrum music, instruments, acoustic speech |
| **`speech`** | **$20$** | $2048$ | $25\,\text{kSPS}$ | $0 - 12.5\,\text{kHz}$ | $81.92\,\text{ms}$ | $12.21\,\text{Hz}$ | Vocal formants, acoustic resonance |
| **`bass_zoom`** | **$50$** | $2048$ | $10\,\text{kSPS}$ | $0 - 5\,\text{kHz}$ | $204.80\,\text{ms}$ | **$4.88\,\text{Hz}$** | Deep sub-bass ($20-100\,\text{Hz}$), room acoustic analysis |

---

## 🖥 Interactive Dashboards

### 1. Dedicated Microphone & Audio Instrument (`ol.audio_dashboard()`)
Designed specifically for passive **MAX4466 electret microphones** (or any analog audio sensor on pins **A0** and **A1**), running completely independently without requiring an Analog Discovery 3:
* **$40.96\,\text{ms} - 204.8\,\text{ms}$ Audio Timebase:** Displays multi-cycle acoustic waveforms for speech, musical instruments, and bass frequencies ($20\,\text{Hz} - 250\,\text{Hz}$).
* **Live VU Meters & Clipping Alerts:** Status bar indicators that flash red if either microphone saturates ($V < 0.10\,\text{V}$ or $V > 3.10\,\text{V}$).
* **Sub-Bin Quadratic Peak Pitch Tracking:** Extracts the dominant acoustic fundamental ($f_0$) with $\pm 0.5\,\text{Hz}$ accuracy.
* **Hann-Windowed Spectral Analysis:** Suppresses spectral leakage by $>32\,\text{dB}$ for clean harmonic overtone visualization.

### 2. Laboratory Dual-Channel Scope & Wavegen (`ol.dashboard()`)
Runs concurrent non-blocking dual waveform generation with the Analog Discovery 3 (W1 & W2) and streaming oscilloscope analysis with live 5-period auto-ranging and selectable hardware trigger routing (`CH1 / A0` vs `CH2 / A1`).

| Dual Scope Triggered on A0 | Dual Scope Triggered on A1 |
| :---: | :---: |
| ![Scope Triggered on A0](https://raw.githubusercontent.com/SiririComun/sw-pynq-oscilloscope/main/docs/images/dashboard_dual_scope_a0.png) | ![Scope Triggered on A1](https://raw.githubusercontent.com/SiririComun/sw-pynq-oscilloscope/main/docs/images/dashboard_dual_scope_a1.png) |

| Dual FFT Spectrum Analyzer | Dedicated Channel 1 View |
| :---: | :---: |
| ![Dual FFT Spectrum](https://raw.githubusercontent.com/SiririComun/sw-pynq-oscilloscope/main/docs/images/dashboard_dual_fft.png) | ![Channel 1 View](https://raw.githubusercontent.com/SiririComun/sw-pynq-oscilloscope/main/docs/images/dashboard_ch1_view.png) |

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

### 3. Install Digilent AD3 Drivers (One-Time Setup for Wavegen)
```python
from pynq_oscilloscope import install_ad3_drivers
install_ad3_drivers()
```

---

## 💻 Python API Usage

### 1. Multi-Regime Profile Switching & Capture
```python
from pynq_oscilloscope import check_usb_permissions, OscilloscopeOverlay

check_usb_permissions()

# Load hardware overlay (auto-fetches v1.4.5 bitstream)
ol = OscilloscopeOverlay()

# 1. Switch to Full Audio Mode (50 kSPS, 40.96 ms frame)
ol.set_profile("audio")
v_a0, v_a1 = ol.capture_stereo()
print(f"Captured Audio: A0 Vpp={v_a0.max()-v_a0.min():.2f}V | A1 Vpp={v_a1.max()-v_a1.min():.2f}V")

# 2. Switch to Deep Bass Zoom Mode (10 kSPS, Δf = 4.88 Hz, 204.8 ms frame)
ol.set_profile("bass_zoom")
freqs, mags = ol.capture_fft(unit="dBV")

# 3. Switch back to High-Speed Lab Scope (500 kSPS, 0 - 250 kHz)
ol.set_profile("oscilloscope")
```

### 2. Jupyter Audio Playback
```python
# Record and listen to microphone audio directly in Jupyter Notebook
ol.set_profile("audio")
ol.play_audio(channel=1) # Plays Channel 1 (A0)
```

### 3. Launch Interactive Dashboards
```python
# Launch dedicated passive Microphone Instrument:
app = ol.audio_dashboard()

# Or launch general Laboratory Scope & AD3 Wavegen Dashboard:
app = ol.dashboard()
```

---

## 📓 Notebook Suite

| Notebook | Description | Key Modules Used |
| :--- | :--- | :--- |
| **`01_ad3_getting_started.ipynb`** | Verifies Digilent drivers and generates analog waveforms in background worker. | `AD3SignalGenerator`, `check_usb_permissions` |
| **`02_xadc_getting_started.ipynb`** | Single-channel hardware triggering and DMA capture on A0. | `OscilloscopeOverlay`, `HardwareTrigger` |
| **`03_oscilloscope_dashboard.ipynb`** | **Laboratory Instrument:** Deploys interactive Dual-Channel AD3 Oscilloscope Dashboard. | `OscilloscopeOverlay`, `OscilloscopeDashboard` |
| **`04_fft_spectrum_analyzer.ipynb`** | **Spectrum Analyzer Guide:** Captures PL hardware FFT spectra, analyzes harmonics. | `OscilloscopeOverlay`, `StreamingFFT` |
| **`05_audio_dashboard.ipynb`** | **Audio Instrument:** Deploys dedicated Audio & Microphone Dashboard (`ol.audio_dashboard()`). | `OscilloscopeOverlay`, `AudioDashboard` |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/SiririComun/sw-pynq-oscilloscope/blob/main/LICENSE) file for details.