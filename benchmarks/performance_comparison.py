"""
benchmarks/performance_comparison.py

Практический бенчмарк для оценки пропускной способности IPC.
Работает даже в минимальном окружении без torch/numpy.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import statistics
import time
from multiprocessing import shared_memory

try:
    import torch  # type: ignore
except Exception:
    torch = None


def _bytes_to_gbps(size_bytes: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return (size_bytes / seconds) / (1024 ** 3)


def benchmark_pipe_pickle(size_mb: int, repeats: int) -> dict:
    """Передача bytes через multiprocessing.Pipe (имитация copy+pickle пути)."""
    payload_size = size_mb * 1024 * 1024
    payload = os.urandom(payload_size)
    timings = []

    def receiver(conn, q):
        started = time.perf_counter()
        data = conn.recv()
        elapsed = time.perf_counter() - started
        q.put((elapsed, len(data)))

    for _ in range(repeats):
        recv_conn, send_conn = mp.Pipe(duplex=False)
        q = mp.Queue()
        proc = mp.Process(target=receiver, args=(recv_conn, q))
        proc.start()

        start = time.perf_counter()
        send_conn.send(payload)
        send_elapsed = time.perf_counter() - start

        proc.join(timeout=120)
        if proc.exitcode != 0:
            raise RuntimeError("Receiver process failed in pipe benchmark")

        recv_elapsed, recv_len = q.get(timeout=5)
        assert recv_len == payload_size
        timings.append(max(send_elapsed, recv_elapsed))

    avg_s = statistics.mean(timings)
    return {
        "name": "Pipe + pickle(bytes)",
        "size_mb": size_mb,
        "repeats": repeats,
        "avg_seconds": avg_s,
        "throughput_gbps": _bytes_to_gbps(payload_size, avg_s),
    }


def benchmark_shared_memory(size_mb: int, repeats: int) -> dict:
    """Передача через shared_memory с чтением без дополнительного копирования."""
    size_bytes = size_mb * 1024 * 1024
    timings = []

    shm = shared_memory.SharedMemory(create=True, size=size_bytes)
    try:
        shm.buf[:] = b"\x2A" * size_bytes

        for _ in range(repeats):
            start = time.perf_counter()
            # Симуляция consumer: открытие, чтение, закрытие.
            view = shm.buf
            checksum = view[0] ^ view[size_bytes // 2] ^ view[-1]
            if checksum != 42:
                raise RuntimeError("Shared memory integrity check failed")
            elapsed = time.perf_counter() - start
            timings.append(elapsed)
    finally:
        shm.close()
        shm.unlink()

    avg_s = statistics.mean(timings)
    return {
        "name": "SharedMemory zero-copy read",
        "size_mb": size_mb,
        "repeats": repeats,
        "avg_seconds": avg_s,
        "throughput_gbps": _bytes_to_gbps(size_bytes, avg_s),
    }


def benchmark_torch_cuda_copy(size_mb: int, repeats: int) -> dict | None:
    """CPU->GPU->CPU roundtrip, если torch+CUDA доступны."""
    if torch is None or not torch.cuda.is_available():
        return None

    size_elements = (size_mb * 1024 * 1024) // 4
    timings = []
    for _ in range(repeats):
        cpu_tensor = torch.randn(size_elements, dtype=torch.float32)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        gpu_tensor = cpu_tensor.cuda(non_blocking=False)
        roundtrip = gpu_tensor.cpu()
        _ = float(roundtrip[0])
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)

    avg_s = statistics.mean(timings)
    return {
        "name": "PyTorch CUDA roundtrip",
        "size_mb": size_mb,
        "repeats": repeats,
        "avg_seconds": avg_s,
        "throughput_gbps": _bytes_to_gbps(size_mb * 1024 * 1024, avg_s),
    }


def run_benchmarks(size_mb: int, repeats: int) -> list[dict]:
    results = [
        benchmark_pipe_pickle(size_mb=size_mb, repeats=repeats),
        benchmark_shared_memory(size_mb=size_mb, repeats=repeats),
    ]
    cuda_result = benchmark_torch_cuda_copy(size_mb=size_mb, repeats=repeats)
    if cuda_result is not None:
        results.append(cuda_result)
    return results


def print_results(results: list[dict]) -> None:
    print("=== ZeroLink v2.0 IPC Performance Benchmarks ===")
    for item in results:
        print(f"\n{item['name']}")
        print(f"  Size: {item['size_mb']} MB, repeats: {item['repeats']}")
        print(f"  Avg time: {item['avg_seconds']:.6f} s")
        print(f"  Throughput: {item['throughput_gbps']:.3f} GiB/s")

    if not any(r["name"] == "PyTorch CUDA roundtrip" for r in results):
        print("\n[info] PyTorch CUDA benchmark skipped: torch and/or CUDA is unavailable.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ZeroLink microbenchmarks")
    parser.add_argument("--size-mb", type=int, default=32, help="Payload size in MB")
    parser.add_argument("--repeats", type=int, default=5, help="Number of repeats")
    args = parser.parse_args()

    results = run_benchmarks(size_mb=args.size_mb, repeats=args.repeats)
    print_results(results)


if __name__ == "__main__":
    main()
