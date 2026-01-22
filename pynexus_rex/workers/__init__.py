"""
zerolink/workers

Workers (CPU и GPU) для распределенного выполнения задач.
"""

from .gpu_worker import GPUWorker
from .cpu_worker import CPUWorker

__all__ = ["GPUWorker", "CPUWorker"]