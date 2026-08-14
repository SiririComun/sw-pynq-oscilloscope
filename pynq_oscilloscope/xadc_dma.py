import numpy as np
from pynq import allocate

class StreamingXADC:
    """
    High-level driver for 1 MSPS XADC AXI-Stream & AXI DMA data acquisition.
    Dynamically binds to the DMA block via .hwh metadata without hardcoded addresses.
    """

    def __init__(self, overlay, default_packet_size: int = 2048):
        """
        Initialize the DMA controller from the loaded PYNQ overlay.
        Default packet size is set to 16,384 to match the hardware packetizer.
        """
        if hasattr(overlay, "axi_dma_0"):
            self.dma = overlay.axi_dma_0
        else:
            dma_blocks = [ip for ip, details in overlay.ip_dict.items() if "dma" in ip.lower()]
            if not dma_blocks:
                raise RuntimeError("No AXI DMA block found in the loaded hardware overlay.")
            self.dma = getattr(overlay, dma_blocks[0])

        self.packet_size = default_packet_size
        # Allocate contiguous CMA buffer ONCE outside capture loop
        self._buffer = allocate(shape=(self.packet_size,), dtype="u2")

    def capture(self, crop_startup_samples: int = 0) -> np.ndarray:
        """
        Triggers a high-speed hardware DMA transfer (S2MM channel),
        waits for completion, scales raw 12-bit left-aligned data to 0.0V - 3.3V,
        and returns a NumPy array of physical voltages.
        """
        # 1. Command DMA to receive incoming stream into CMA RAM
        self.dma.recvchannel.transfer(self._buffer)
        
        # 2. Block until hardware transfer completes (asserted by TLAST at packet end)
        self.dma.recvchannel.wait()
        
        # 3. Cast to NumPy array
        raw_samples = np.array(self._buffer)
        
        # 4. Scale 12-bit left-aligned data (shift 4 right) to 0V - 3.3V
        voltages = (raw_samples >> 4) * (3.3 / 4095.0)
        
        if crop_startup_samples > 0 and len(voltages) > crop_startup_samples:
            voltages = voltages[crop_startup_samples:]
            
        return voltages

    def close(self):
        """Free contiguous memory allocation (CMA) buffer."""
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