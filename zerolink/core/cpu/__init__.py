"""
zerolink/core/cpu

CPU Subsystem (Shared Memory).
Обеспечивает Zero-Copy обмен данными между процессами на CPU.
"""

from .shm_pool import SharedMemoryPool, CPUBlock

__all__ = ["SharedMemoryPool", "CPUBlock"]