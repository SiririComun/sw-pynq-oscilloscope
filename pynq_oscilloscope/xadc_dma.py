from typing import Tuple
import numpy as np
from pynq import allocate


class StreamingXADC:
    """
    High-level driver for XADC AXI-Stream & AXI DMA data acquisition.
    Supports Single-Channel (A0) and Simultaneous Dual-Channel (A0 & A1) streaming.
    """

    def __init__(self, overlay, default_packet_size: int = 2048):
        if hasattr(overlay, "axi_dma_0"):
            self.dma = overlay.axi_dma_0
        else:
            dma_blocks = [ip for ip, details in overlay.ip_dict.items() if "dma" in ip.lower()]
            if not dma_blocks:
                raise RuntimeError("No AXI DMA block found in the loaded hardware overlay.")
            self.dma = getattr(overlay, dma_blocks[0])

        self.packet_size = default_packet_size
        self._buffer = allocate(shape=(self.packet_size,), dtype="u2")

    def capture_stereo(self, crop_startup_samples: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Captures simultaneous interleaved stereo samples from XADC via DMA.
        
        :return: (voltages_ch1_a0, voltages_ch2_a1) arrays of length packet_size / 2.
        """
        # 1. Trigger DMA transfer
        self.dma.recvchannel.transfer(self._buffer)
        self.dma.recvchannel.wait()
        
        raw_samples = np.array(self._buffer)
        
        # 2. De-interleave: Even = Channel 1 (A0 / Vaux1), Odd = Channel 2 (A1 / Vaux9)
        raw_ch1 = raw_samples[0::2]
        raw_ch2 = raw_samples[1::2]
        
        # 3. Scale 12-bit left-aligned data (shift 4 right) to 0.0V - 3.3V
        voltages_ch1 = (raw_ch1 >> 4) * (3.3 / 4095.0)
        voltages_ch2 = (raw_ch2 >> 4) * (3.3 / 4095.0)
        
        if crop_startup_samples > 0:
            voltages_ch1 = voltages_ch1[crop_startup_samples:]
            voltages_ch2 = voltages_ch2[crop_startup_samples:]
            
        return voltages_ch1, voltages_ch2

    def capture(self, crop_startup_samples: int = 0) -> np.ndarray:
        """Backwards-compatible single-channel capture (returns Channel 1 on A0)."""
        v_ch1, _ = self.capture_stereo(crop_startup_samples=crop_startup_samples)
        return v_ch1

    def close(self):
        """Free allocated contiguous memory (CMA) buffer."""
        if hasattr(self, "_buffer") and self._buffer is not None:
            try:
                self._buffer.close()
                self._buffer = None
            except Exception:
                pass

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()