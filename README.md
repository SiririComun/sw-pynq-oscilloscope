# Real-Time 1 MSPS Hardware-Triggered PYNQ Oscilloscope

[![PyPI Version](https://img.shields.io/pypi/v/pynq-oscilloscope.svg)](https://pypi.org/project/pynq-oscilloscope/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hardware Overlay](https://img.shields.io/badge/Hardware-hw--xadc--dma--overlays%20v1.1.0--rc1-orange.svg)](https://github.com/SiririComun/hw-xadc-dma-overlays)
[![Board Support](https://img.shields.io/badge/Board-PYNQ--Z2-green.svg)](https://tul.com.tw/ProductsPYNQ-Z2.html)

A high-performance, dark-mode real-time Oscilloscope software stack running natively on PYNQ Linux platforms. 

Features **sub-microsecond hardware-level edge triggering** (`axis_trigger_unit`), **1 MSPS XADC streaming via AXI DMA** direct to DDR memory, and non-blocking analog signal generation with the **Digilent Analog Discovery 3** via `pydwf`.

---

## 🏛 System Architecture

This repository adopts the **canonical PYNQ Custom Overlay pattern** (`OscilloscopeOverlay`). It automatically pulls its compiled hardware bitstream and metadata from GitHub Releases (or loads local custom `.bit` builds) and encapsulates the DMA receiver, AXI-Lite trigger registers, and wavegen into a unified Python object.

```
 [ Analog Discovery 3 (W1) ] ──(Analog Jumper Wire)──> [ PYNQ-Z2 Header (A0) ]
              │                                                     │
        (pydwf SDK)                                     (XADC 1 MSPS AXI-Stream)
              │                                                     │
              ▼                                                     ▼
 [ AD3SignalGenerator ]                              [ axis_trigger_unit IP ]
              │                                     (Edge, Threshold, Auto Timeout)
              │                                                     │
              │                                                     ▼
              │                                          [ AXI DMA S2MM Engine ]
              │                                                     │
              └───────────────────────┬─────────────────────────────┘
                                      │
                                      ▼
                           [ OscilloscopeOverlay ]
                   (Subclasses pynq.Overlay with sub-drivers)
                    ├── .trigger  (HardwareTrigger AXI-Lite)
                    ├── .xadc     (StreamingXADC DMA Driver)
                    ├── .wavegen  (AD3SignalGenerator)
                    └── .dashboard() (Interactive Plotly Canvas)
```

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
Connect to your PYNQ board via SSH or Jupyter Terminal and run:

```bash
pip install --upgrade pynq-oscilloscope
```

### 2. Copy Example Notebooks to Jupyter Workspace
Copy this project's notebooks into `/home/xilinx/jupyter_notebooks/pynq_oscilloscope/`:

```bash
pynq-oscilloscope-get-notebooks
```

### 3. Install Digilent AD3 Drivers
Run the automated environment setup inside Python or a Jupyter cell:

```python
from pynq_oscilloscope import install_ad3_drivers

# Downloads Adept Runtime + WaveForms SDK and configures USB permissions
install_ad3_drivers()
```

---

## 💻 Python API Usage

### 1. Launch Interactive Dashboard in 2 Lines (Default Cloud Fetch)
```python
from pynq_oscilloscope import OscilloscopeOverlay

# Automatically identifies board (PYNQ-Z2), downloads v1.1.0-rc1 release, and loads FPGA
ol = OscilloscopeOverlay()

# Launch dark-mode interactive Plotly + IPywidgets dashboard
app = ol.dashboard()
```

### 2. Load Local Custom Bitstream (Offline / Development)
```python
from pynq_oscilloscope import OscilloscopeOverlay

# Load a local bitstream while preserving all driver hooks and UI tools
ol = OscilloscopeOverlay("./pynq_z2.bit")
app = ol.dashboard()
```

### 3. Programmatic Hardware Trigger & DMA Capture
```python
from pynq_oscilloscope import OscilloscopeOverlay

ol = OscilloscopeOverlay()

# Configure FPGA Trigger: Rising Edge @ 1.65V with 50 ms Auto-timeout
ol.trigger.configure(mode="Auto", edge="Rising", threshold_volts=1.65, timeout_ms=50.0)

# Capture 16,384 samples (Sample [0] is guaranteed hardware-aligned to trigger point!)
voltages = ol.capture()
print(f"Captured {len(voltages)} samples. Min: {voltages.min():.2f}V, Max: {voltages.max():.2f}V")

# Clean release of memory buffers
ol.close()
```

---

## 📓 Notebook Suite

| Notebook | Description | Key Modules Used |
| :--- | :--- | :--- |
| **`01_ad3_getting_started.ipynb`** | Verifies Digilent drivers and generates analog signals (Sine, Square, Triangle) in a non-blocking background worker. | `AD3SignalGenerator`, `check_usb_permissions` |
| **`02_xadc_getting_started.ipynb`** | Demonstrates `OscilloscopeOverlay`, hardware register trigger configuration (`ol.trigger`), and DMA capture. | `OscilloscopeOverlay`, `HardwareTrigger` |
| **`03_oscilloscope_dashboard.ipynb`** | **Main Application:** Deploys the complete interactive Plotly Oscilloscope with live trigger line, auto-ranging, and AD3 integration. | `OscilloscopeOverlay` |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.