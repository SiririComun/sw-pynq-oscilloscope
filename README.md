# Real-Time 1 MSPS PYNQ Oscilloscope

[![PyPI Version](https://img.shields.io/pypi/v/pynq-oscilloscope.svg)](https://pypi.org/project/pynq-oscilloscope/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hardware Overlay](https://img.shields.io/badge/Hardware-hw--xadc--dma--overlays%20v1.0.2-orange.svg)](https://github.com/SiririComun/hw-xadc-dma-overlays)
[![Board Support](https://img.shields.io/badge/Board-PYNQ--Z2-green.svg)](https://tul.com.tw/ProductsPYNQ-Z2.html)

A high-performance, interactive, dark-mode real-time Oscilloscope software stack running natively on PYNQ Linux platforms. 

Combines high-speed FPGA data acquisition (**1 MSPS XADC streaming via AXI DMA**) with an active analog wave generator (**Digilent Analog Discovery 3** via `pydwf`) into an interactive Plotly + IPywidgets dashboard.

---

## 🏛 System Architecture

This software repository operates as a lightweight client. It **automatically fetches its compiled hardware overlay binaries** (`pynq_z2.bit` and `pynq_z2.hwh`) from the pinned release `v1.0.2` of the [hw-xadc-dma-overlays](https://github.com/SiririComun/hw-xadc-dma-overlays) repository.

```
 [ Analog Discovery 3 (W1) ] ──(Analog Jumper Wire)──> [ PYNQ-Z2 Header (A0) ]
              │                                                     │
        (pydwf SDK)                                           (AXI DMA 1 MSPS)
              │                                                     │
              ▼                                                     ▼
 [ AD3SignalGenerator ] <──(pynq_oscilloscope)──> [ StreamingXADC DMA Driver ]
                                      │
                                      ▼
                        [ OscilloscopeDashboard ]
                 (Interactive Dark-Mode Plotly UI Canvas)
```

---

## 🔌 Hardware Setup & Prerequisites

Before running the application, make sure your hardware is connected according to these physical specifications:

1. **USB Port Connection:**
   * Plug the Analog Discovery 3 USB cable into the large rectangular **USB HOST** port on the PYNQ-Z2 board (next to the Ethernet port).
2. **USB Cable Quality:**
   * Use a high-quality **Data + Power USB-C cable**. Standard charging-only cables omit data lines.
3. **Power Supply:**
   * Power the AD3 with an external **5V auxiliary power supply** to prevent board brownouts under load.
4. **Signal Wire:**
   * Connect a jumper wire from **Wavegen 1 (W1)** on the AD3 to **Analog Input A0** on the PYNQ-Z2 shield header. Connect AD3 **GND** to PYNQ **GND**.

---

## 🚀 Quick Start & Installation

### 1. Install Package from PyPI
Connect to your PYNQ board via SSH or Jupyter Terminal and run:

```bash
pip install pynq-oscilloscope
```

### 2. Copy Example Notebooks to Jupyter Workspace
To copy this project's notebooks into a dedicated subfolder (`/home/xilinx/jupyter_notebooks/pynq_oscilloscope/`) without touching other installed PYNQ packages, run:

```bash
pynq-oscilloscope-get-notebooks
```

*Alternatively, inside a Python or Jupyter session:*
```python
from pynq_oscilloscope import copy_notebooks

copy_notebooks()
```

### 3. Install Digilent AD3 Drivers
Run the automated environment checker inside Python or Jupyter:

```python
from pynq_oscilloscope import install_ad3_drivers

# Automatically downloads Digilent Adept + WaveForms .deb packages and sets USB permissions
install_ad3_drivers()
```

---

## 🧪 Isolated Virtual Environment Setup Guide (Optional)

If you want to test or run `pynq-oscilloscope` inside an isolated Python virtual environment on your PYNQ board:

```bash
# 1. Create a virtual environment with system site-packages enabled
python3 -m venv --system-site-packages /home/xilinx/clean_test_env
source /home/xilinx/clean_test_env/bin/activate

# 2. Link PYNQ system driver site-packages
echo "/usr/local/share/pynq-venv/lib/python3.10/site-packages" > /home/xilinx/clean_test_env/lib/python3.10/site-packages/pynq_system.pth

# 3. Install pynq-oscilloscope from PyPI
pip install --no-deps -I pynq-oscilloscope

# 4. Copy notebooks & register Jupyter kernel
pynq-oscilloscope-get-notebooks
pip install ipykernel
python -m ipykernel install --user --name=clean_test_env --display-name "Python 3 (Clean Test Env)"
```

In Jupyter Notebook:
1. Navigate to `pynq_oscilloscope/` and open `03_oscilloscope_dashboard.ipynb`.
2. Click **Kernel** $\rightarrow$ **Change kernel** $\rightarrow$ **`Python 3 (Clean Test Env)`**.
3. Run the cells!

---

## 📓 Notebook Suite

This repository includes three progressive interactive notebooks inside the `notebooks/` directory:

| Notebook | Description | Key Modules Used |
| :--- | :--- | :--- |
| **`01_ad3_getting_started.ipynb`** | Verifies Digilent drivers and generates analog signals (Sine, Square, Triangle) in a background thread. | `AD3SignalGenerator`, `check_usb_permissions` |
| **`02_xadc_getting_started.ipynb`** | Automatically fetches `v1.0.2` overlay and captures 1 MSPS analog streams direct to DDR memory. | `HardwareLoader`, `StreamingXADC` |
| **`03_oscilloscope_dashboard.ipynb`** | **Main Application:** Deploys the complete interactive closed-loop Plotly Oscilloscope with triggers and auto-ranging. | `OscilloscopeDashboard` |

---

## 💻 Python Package Usage Example

You can deploy the complete Oscilloscope Dashboard in 3 lines of Python code:

```python
from pynq_oscilloscope import HardwareLoader, OscilloscopeDashboard

# 1. Fetch board overlay (v1.0.2) from GitHub Releases
overlay = HardwareLoader.load_overlay()

# 2. Instantiate and render interactive Oscilloscope
app = OscilloscopeDashboard(overlay=overlay)
app.display()
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
