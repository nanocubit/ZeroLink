"""
ZeroLink v2.0
Enterprise-Ready GPU Memory Runtime for Python.
"""

__version__ = "2.0.0"

# Public API
try:
    from .runtime.unified import UnifiedRuntime
    from .monitoring import telemetry, prometheus_exporter
    from .profiling import profiler
    from .core.gpu import PinnedMemoryManager, MultiDevicePool

    __all__ = [
        "UnifiedRuntime",
        "telemetry",
        "prometheus_exporter",
        "profiler",
        "PinnedMemoryManager",
        "MultiDevicePool"
    ]
except ImportError:
    # Если runtime не готов (на этапе разработки), оставляем пустым
    __all__ = []