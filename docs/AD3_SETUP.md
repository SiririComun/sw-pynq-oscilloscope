# Digilent Analog Discovery 3 (AD3) Setup Guide for PYNQ

This guide provides hardware connection instructions, driver installation steps, and USB permission configurations to run the Digilent WaveForms SDK natively on PYNQ Linux operating systems.

---

## ⚠️ Critical Hardware Compatibility Warnings

Before attempting driver installation or software execution, verify your hardware connections:

1. **USB Port Connection:**
   * The AD3's USB cable **must** be physically connected to the large, rectangular **USB HOST** port on the PYNQ board (located next to the Ethernet jack).
   * Do **not** plug the AD3 into the Micro-USB ports; those are strictly for powering and programming the PYNQ board itself.

2. **USB Cable Quality:**
   * You **must** use a high-quality **Data + Power USB-C cable**.
   * Standard phone charging-only cables omit internal copper data lines, rendering the AD3 invisible to Linux.

3. **External Power Supply:**
   * To prevent board brownouts under heavy processing loads, power your Analog Discovery 3 using its external **5V auxiliary power supply**.

---

## 🚀 Automated Driver Installation (Recommended)

The `pynq_oscilloscope` package includes an architecture-aware environment installer that automatically detects your board's CPU (32-bit `armhf` vs. 64-bit `aarch64`), downloads the matching Digilent Debian packages, and sets USB bus permissions.

Run the following inside a Python environment or Jupyter notebook cell:

```python
from pynq_oscilloscope import install_ad3_drivers

# Download Adept Runtime + WaveForms SDK and set USB permissions
install_ad3_drivers()
```

---

## 🛠 Manual Driver Installation (Debian Packages)

If you prefer installing the drivers manually, execute the commands below in your board's Linux terminal.

### For 32-bit ARM Boards (PYNQ-Z2 / PYNQ-Z1 / ZedBoard — `armhf`)
```bash
# 1. Download Digilent Adept Runtime and WaveForms SDK
wget -q https://files.digilent.com/Software/Adept2%20Runtime/2.27.9/digilent.adept.runtime_2.27.9-armhf.deb
wget -q -O digilent.waveforms_armhf.deb https://files.digilent.com/Software/Waveforms/3.25.1/digilent.waveforms_qt5_3.25.1_armhf.deb

# 2. Install Debian packages
sudo dpkg -i digilent.adept.runtime_2.27.9-armhf.deb
sudo dpkg -i digilent.waveforms_armhf.deb

# 3. Resolve missing Qt5 dependencies
sudo apt-get install -f -y

# 4. Install Python wrapper
pip install pydwf
```

### For 64-bit ARM Boards (Kria KV260 / KR260 — `aarch64`)
```bash
# 1. Download Digilent Adept Runtime and WaveForms SDK for 64-bit
wget -q https://files.digilent.com/Software/Adept2%20Runtime/2.27.9/digilent.adept.runtime_2.27.9-arm64.deb
wget -q -O digilent.waveforms_arm64.deb https://files.digilent.com/Software/Waveforms/3.25.1/digilent.waveforms_qt5_3.25.1_arm64.deb

# 2. Install Debian packages
sudo dpkg -i digilent.adept.runtime_2.27.9-arm64.deb
sudo dpkg -i digilent.waveforms_arm64.deb

# 3. Resolve missing dependencies
sudo apt-get install -f -y

# 4. Install Python wrapper
pip install pydwf
```

---

## 🔒 Managing Linux USB Permissions

By default, Linux blocks unprivileged users (such as the `xilinx` user in Jupyter) from communicating directly with raw USB devices.

To grant immediate access without rebooting the board:

```bash
sudo chmod -R 777 /dev/bus/usb/
```

Or execute the Python helper inside your notebook:
```python
from pynq_oscilloscope import check_usb_permissions

check_usb_permissions()
```

---

## 🧪 Verifying Connection in Python

Run this verification block to confirm that the PYNQ board can communicate with your AD3:

```python
from pydwf import DwfLibrary
from pydwf.utilities import openDwfDevice

dwf = DwfLibrary()

try:
    with openDwfDevice(dwf) as device:
        print("=====================================================")
        print(" SUCCESS: Your Analog Discovery 3 is connected!")
        print("=====================================================")
except Exception as e:
    print(f"CONNECTION FAILED: {e}")
```