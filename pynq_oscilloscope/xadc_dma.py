"""
pynq_oscilloscope.xadc_dma: High-Performance AXI DMA Stream Receiver.
Supports single-frame capture, stereo de-interleaving, and double-buffered
continuous multi-frame streaming.
"""

from typing import Tuple, Optional
import time
import numpy as np
from pynq import allocate


class StreamingXADC:
    """
    High-level driver for XADC AXI4-Stream & AXI DMA data acquisition.
    Dynamically binds to the DMA engine via .hwh metadata.
    """

    def __init__(self, overlay, default_packet_size: int = 2048):
        """
        Initialize DMA controller and pre-allocate double buffers.
        """
        if hasattr(overlay, "axi_dma_0"):
            self.dma = overlay.axi_dma_0
        else:
            dma_blocks = [ip for ip, details in overlay.ip_dict.items() if "dma" in ip.lower()]
            if not dma_blocks:
                raise RuntimeError("No AXI DMA block found in the loaded hardware overlay.")
            self.dma = getattr(overlay, dma_blocks[0])

        self.packet_size = default_packet_size

        # Allocate primary CMA buffer and double-buffers (Ping-Pong)
        self._buffer = allocate(shape=(self.packet_size,), dtype="u2")
        self._buf_a = allocate(shape=(self.packet_size,), dtype="u2")
        self._buf_b = allocate(shape=(self.packet_size,), dtype="u2")

    def capture_raw(self) -> np.ndarray:
        """
        Captures a single raw frame (packet_size words) direct from hardware.
        """
        self.dma.recvchannel.transfer(self._buffer)
        self.dma.recvchannel.wait()
        return np.array(self._buffer, copy=True)

    def capture(self, crop_startup_samples: int = 0) -> np.ndarray:
        """
        Captures a single frame scaled to physical voltages (0.0V - 3.3V).
        """
        raw_samples = self.capture_raw()
        voltages = (raw_samples >> 4) * (3.3 / 4095.0)

        if crop_startup_samples > 0 and len(voltages) > crop_startup_samples:
            voltages = voltages[crop_startup_samples:]

        return voltages

    def capture_stereo(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Captures a single interleaved frame and splits into Channel 1 (A0) and Channel 2 (A1).
        """
        raw_interleaved = self.capture_raw()
        raw_ch1 = raw_interleaved[0::2]
        raw_ch2 = raw_interleaved[1::2]

        voltages_ch1 = (raw_ch1 >> 4) * (3.3 / 4095.0)
        voltages_ch2 = (raw_ch2 >> 4) * (3.3 / 4095.0)

        return voltages_ch1, voltages_ch2

    def capture_continuous_raw(
        self,
        num_frames: int,
        trigger_unit=None,
        timeout_sec: float = 5.0
    ) -> np.ndarray:
        """
        Performs continuous, gapless multi-frame DMA streaming using Ping-Pong buffers.

        :param num_frames: Total number of sequential 2048-sample frames to capture.
        :param trigger_unit: Optional HardwareTrigger instance to ensure Auto Mode arming.
        :param timeout_sec: Per-frame timeout in seconds before aborting.
        :return: 1D NumPy array of shape (num_frames * packet_size,), dtype=uint16.
        """
        if num_frames <= 0:
            return np.empty(0, dtype=np.uint16)

        total_samples = num_frames * self.packet_size
        out_data = np.empty(total_samples, dtype=np.uint16)

        # 1. Reset DMA channel to clear any stale state
        try:
            self.dma.mmio.write(0x30, 0x04) # S2MM_DMACR.Reset = 1
            time.sleep(0.002)
            self.dma.recvchannel.start()
        except Exception:
            pass

        # 2. Ensure trigger unit is configured for streaming
        if trigger_unit is not None:
            # Force Arm in Auto Mode (Bit 0=1, Bit 1=1)
            ctrl = trigger_unit.mmio.read(0x00)
            trigger_unit.mmio.write(0x00, ctrl | 0x03)

        # 3. Prime the first buffer (Buffer A)
        active_buf = self._buf_a
        next_buf = self._buf_b
        self.dma.recvchannel.transfer(active_buf)

        # 4. Double-buffered chaining loop
        for frame_idx in range(num_frames):
            t_start = time.time()

            # Wait for currently transferring buffer
            while not self.dma.recvchannel.idle:
                if time.time() - t_start > timeout_sec:
                    # Safe hardware abort on timeout
                    self.dma.mmio.write(0x30, 0x04)
                    raise TimeoutError(
                        f"[StreamingXADC] DMA transfer timed out at frame {frame_idx + 1}/{num_frames}."
                    )
                time.sleep(0.0005)

            # Immediately queue the NEXT buffer (< 5 µs latency) before copying
            if frame_idx + 1 < num_frames:
                self.dma.recvchannel.transfer(next_buf)

            # Copy completed buffer data into main destination RAM
            offset = frame_idx * self.packet_size
            out_data[offset : offset + self.packet_size] = np.array(active_buf, copy=False)

            # Swap Ping-Pong buffer pointers
            active_buf, next_buf = next_buf, active_buf

        return out_data

    def close(self):
        """Free allocated contiguous memory (CMA) buffers."""
        for buf_attr in ["_buffer", "_buf_a", "_buf_b"]:
            if hasattr(self, buf_attr) and getattr(self, buf_attr) is not None:
                try:
                    getattr(self, buf_attr).close()
                    setattr(self, buf_attr, None)
                except Exception:
                    pass

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()