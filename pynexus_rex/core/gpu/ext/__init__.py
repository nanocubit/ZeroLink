"""
zerolink/core/gpu/ext

Wrapper for the CUDA VMM C++ Extension (ipc_ext).
Загружает бинарный модуль и экспортирует функции API.
"""

# Импорт скомпилированного C++ модуля
# setup.py компилирует ipc_ext.cpp именно в этот модуль
try:
    from . import ipc_ext
except ImportError as e:
    # Перехватываем ImportError для предоставления понятного сообщения пользователю
    raise ImportError(
        "Не удалось загрузить C++ расширение 'zerolink.core.gpu.ext.ipc_ext'. "
        "Убедитесь, что:\n"
        "1. Установлены CUDA Drivers (CUDA 11.2+).\n"
        "2. Выполнена сборка расширения: 'python setup.py build_ext --inplace'.\n"
        f"Исходная ошибка: {e}"
    ) from e

# Экспорт функций на уровень пакета для удобства использования
# Пользователь может писать: from zerolink.core.gpu.ext import get_granularity
from .ipc_ext import (
    get_granularity,
    import_vmm_segments,
    tensor_from_imported,
    ptr
)

__all__ = [
    "get_granularity",
    "import_vmm_segments",
    "tensor_from_imported",
    "ptr",
]

__version__ = "2.0.0"