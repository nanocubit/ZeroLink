# Benchmarking Guide / Руководство по бенчмаркам

## RU

### Что проверяет бенчмарк

Скрипт `benchmarks/performance_comparison.py` измеряет IPC-пути:

- `Pipe + pickle(bytes)` — baseline для copy/serialization подхода.
- `SharedMemory zero-copy read` — путь с доступом к данным через shared memory без дополнительного копирования.
- `PyTorch CUDA roundtrip` — опционально, только если доступны `torch` и CUDA.

### Как запускать

```bash
python benchmarks/performance_comparison.py --size-mb 32 --repeats 5
```

Параметры:

- `--size-mb` — размер полезной нагрузки в MB.
- `--repeats` — число повторов для усреднения.

### Интерпретация

- `Avg time` — среднее время одного прогона.
- `Throughput` — пропускная способность в GiB/s.
- Если `PyTorch CUDA roundtrip` пропущен, значит окружение не содержит `torch` и/или CUDA.

---

## EN

### What this benchmark measures

`benchmarks/performance_comparison.py` evaluates multiple IPC data paths:

- `Pipe + pickle(bytes)` — copy/serialization baseline.
- `SharedMemory zero-copy read` — shared-memory path with zero-copy style reads.
- `PyTorch CUDA roundtrip` — optional path, only if `torch` and CUDA are available.

### How to run

```bash
python benchmarks/performance_comparison.py --size-mb 32 --repeats 5
```

Arguments:

- `--size-mb` — payload size in MB.
- `--repeats` — number of repetitions for averaging.

### Result interpretation

- `Avg time` — average runtime per iteration.
- `Throughput` — measured throughput in GiB/s.
- If `PyTorch CUDA roundtrip` is skipped, the environment does not provide `torch` and/or CUDA.
