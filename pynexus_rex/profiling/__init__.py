"""
zerolink/profiling

Модуль профилирования для ZeroLink v2.0.
"""

from .profiler import profiler, Profiler, profile_gpu_function

__all__ = [
    "profiler",
    "Profiler",
    "profile_gpu_function"
]