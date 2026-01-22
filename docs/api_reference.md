# API Reference для ZeroLink v2.0

## Основные классы

### ZeroLinkRuntime (ранее UnifiedRuntime)

Основной класс для управления GPU и CPU пулами.

#### Конструктор

```python
from zerolink.runtime import ZeroLinkRuntime

runtime = ZeroLinkRuntime(
    socket_path: str = "/tmp/zerolink.sock",
    gpu_device_id: int = 0,
    gpu_pool_size_gb: int = 4,
    cpu_pool_name: str = "zerolink_cpu_pool",
    cpu_pool_size_mb: int = 512,
    num_cpu_workers: int = 4,
    enable_integrity_hash: bool = True
)
```

#### Методы

- `start(block: bool = False)` - запускает сервер
- `stop()` - останавливает сервер
- `execute_gpu(func, *args, **kwargs)` - выполнение функции на GPU
- `execute_cpu(func, *args, **kwargs)` - выполнение функции на CPU
- `map_cpu(func, iterable)` - применение функции к итерируемому объекту на CPU
- `get_pool_stats()` - получение статистики пулов
- `get_ipc_lease_status()` - получение статуса IPC аренд

### Вспомогательные классы

- `PinnedMemoryManager` - управление закрепленной памятью
- `MultiDevicePool` - пул для нескольких GPU
- `telemetry` - система телеметрии
- `profiler` - встроенный профилировщик