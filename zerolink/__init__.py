"""
ZeroLink - Unified Zero-Copy Runtime for CPU/GPU Computing
Copyright (C) 2025 Slava Maltsev <nanotec@live.ru>
SPDX-License-Identifier: MIT

ZeroLink v2.0
Enterprise-Ready GPU Memory Runtime for Python.
"""

__version__ = "2.0.0"

# Public API
try:
    from .runtime.unified import ZeroLinkRuntime, UnifiedRuntime
    from .monitoring import telemetry, prometheus_exporter
    from .profiling import profiler
    from .core.gpu import PinnedMemoryManager, MultiDevicePool

    __all__ = [
        "ZeroLinkRuntime",
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