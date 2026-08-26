"""
tests/conftest.py: Universal hardware mocking fixture for cloud CI & PC test runs.
Automatically shims 'pynq' with in-memory MMIO dictionary if running on non-board PC.
"""

import sys
from unittest.mock import MagicMock


class DummyMMIO:
    """In-memory register dictionary emulator matching Xilinx MMIO interface."""
    def __init__(self, base_addr: int = 0, length: int = 65536):
        self.base_addr = base_addr
        self.length = length
        self.regs = {}

    def read(self, offset: int) -> int:
        return self.regs.get(offset, 0)

    def write(self, offset: int, value: int):
        self.regs[offset] = int(value)


# Always ensure pynq.MMIO is available on host PC / CI runner
if "pynq" not in sys.modules:
    try:
        import pynq
        if not hasattr(pynq, "MMIO"):
            pynq.MMIO = DummyMMIO
    except ImportError:
        pynq_mock = MagicMock()
        pynq_mock.MMIO = DummyMMIO
        pynq_mock.allocate = MagicMock()
        pynq_mock.Overlay = MagicMock()
        pynq_mock.Device = MagicMock()
        sys.modules["pynq"] = pynq_mock