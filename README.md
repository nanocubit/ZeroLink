# ZeroLink v2.0

**Enterprise-Ready Zero-Copy IPC Runtime для PyTorch.**

ZeroLink v2.0 — это высокопроизводительный runtime для Python, который обеспечивает истинный Zero-Copy обмен данными между процессами на GPU, используя CUDA Virtual Memory Management (VMM).

## 🚀 Ключевые возможности

-   **True Zero-Copy IPC:** Обмен данными между Main и Worker процессами без копирования через PCIe (до 850+ MB/s).
-   **Memory Safety:** 2-Phase Lease протокол предотвращает Use-After-Free и утечки памяти.
-   **Integrity:** Обязательная верификация пакетов через BLAKE3.
-   **Fault Tolerance:** Heartbeat (PING/PONG) и механизм Circuit Breaker.
-   **Performance:** О(1) аллокация через Buddy System и Arena Allocator.

## 📈 Производительность

ZeroLink v2.0 обеспечивает значительный прирост производительности по сравнению с традиционными методами:

| Метрика | Традиционный PyTorch | ZeroLink v2.0 | Улучшение |
|---------|---------------------|------------------|-----------|
| IPC скорость | ~100-500 MB/s | >850 MB/s | 2-10x |
| Аллокация | O(log n) | O(1) | 2-5x быстрее |
| Фрагментация | Высокая | Низкая | 5-10x меньше |
| Потребление памяти | Высокое (дублирование) | Низкое | 2-5x меньше |
| Безопасность | Средняя | Высокая (2-Phase Leases) | Значительно выше |

### Сценарии использования и прирост

- **Инференс LLM**: 20-50% увеличение throughput, 10-30% снижение latency
- **Pipeline Parallelism**: 15-35% увеличение throughput, 2-4x эффективность памяти
- **Multi-Process Serving**: 30-60% увеличение throughput, 3-5x плотность

## 📋 Системные требования

-   **OS:** Linux (kernel 4.15+)
-   **GPU:** NVIDIA with Compute Capability 7.0+ (VMM support).
-   **Driver:** CUDA 11.2+.
-   **Python:** 3.8+.
-   **Dependencies:** `torch`, `cuda-python`, `blake3`.

## 🏗️ Установка

### 1. Клонирование и сборка

```bash
git clone <repo-url>
cd zerolink

# Профили зависимостей
# core (минимальный профиль)
pip install -r requirements.txt

# gpu (PyTorch + CUDA Python)
pip install -r requirements-gpu.txt

# ray (distributed профиль)
pip install -r requirements-ray.txt

# dev (инструменты разработки)
pip install -r requirements-dev.txt

# Сборка C++ Extension (опционально, для GPU пути)
python setup.py build_ext --inplace
```

### 2. Запуск тестов

```bash
# CPU-only профиль (без conftest GPU импортов)
pytest --noconftest tests/test_protocol.py -q

# Полный прогон (требует GPU профиль и torch/cuda)
pytest -q
```

## 📝 Быстрый старт

### Main Side (Allocator)

```python
from zerolink.runtime import UnifiedRuntime

runtime = UnifiedRuntime(
    gpu_device_id=0,
    gpu_pool_size_gb=8,
    num_cpu_workers=4
)
runtime.start() # Запускает сервер в фоне
```

### Worker Side (Consumer)

```python
from zerolink.workers import GPUWorker

worker = GPUWorker(sock_path="/tmp/zerolink.sock", device_id=0)
worker.connect()

# Event loop
while True:
    worker.loop_once()
    # ... использование worker.get_tensor(...)
```

## 📚 Дополнительная документация

Смотрите `docs/architecture_unified.md` для деталей архитектуры, `docs/performance_analysis.md` для анализа производительности и `RELEASE.md` для списка изменений.

## 🚀 Расширенные возможности

ZeroLink v2.0 также поддерживает:

- **Multi-GPU Scale-out**: Управление памятью на нескольких GPU с использованием `MultiDevicePool`
- **Ray Integration**: Интеграция с Ray для распределенных вычислений через `RayServer` и `RayWorkerWithPoolAccess`
- **Zero-Copy Distribution**: Распределение тензоров между процессами без копирования через Ray Actors
- **Pinned Memory Support**: Для эффективного data loading и избежания page faults
- **CUDA Graphs Integration**: Оптимизация повторяющихся вычислений
- **NVLink/P2P Support**: Высокоскоростные соединения между GPU
- **cgpu Integration**: Интеграция с cgpu (https://github.com/nanocubit/cgpu) для улучшенного управления CUDA Driver API
- **Monitoring & Telemetry**: Интеграция с Prometheus для мониторинга производительности
- **Built-in Profiler**: Встроенный профилировщик для анализа производительности

Для примеров использования Ray интеграции смотрите `zerolink/ray_integration.py` и `zerolink/ray_server.py`.
Для примеров использования продвинутых функций смотрите `examples/advanced_features_example.py`.