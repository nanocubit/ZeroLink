# Интеграция с Ray в ZeroLink v2.0

## Обзор

ZeroLink v2.0 предоставляет интеграцию с Ray для распределенных вычислений и масштабирования.

## Использование

```python
from zerolink.ray_server import RayServer, GlobalPoolConfig
from zerolink.ray_integration import RayClusterManager, RayWorkerConfig

# Настройка Ray сервера
config = GlobalPoolConfig(
    gpu_device_ids=[0, 1],
    gpu_pool_size_gb=8,
    num_cpu_workers=4
)

server = RayServer(config)
```

## Особенности

- Поддержка распределенного пула памяти
- Zero-Copy передача тензоров между Ray Actor'ами
- Интеграция с системой мониторинга ZeroLink