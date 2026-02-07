"""
ZeroLink - Unified Zero-Copy Runtime for CPU/GPU Computing
Copyright (C) 2025 Slava Maltsev <nanotec@live.ru>
SPDX-License-Identifier: MIT

zerolink/monitoring/telemetry.py

Система мониторинга и телеметрии для ZeroLink v2.0.
"""

import time
import threading
from dataclasses import dataclass
from typing import Dict, Callable, Optional
from enum import Enum


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass
class Metric:
    """Базовая метрика."""

    name: str
    type: MetricType
    value: float = 0.0
    labels: Optional[Dict[str, str]] = None
    description: str = ""

    def __post_init__(self):
        if self.labels is None:
            self.labels = {}


class TelemetryCollector:
    """Сборщик телеметрии."""

    def __init__(self):
        self.metrics: Dict[str, Metric] = {}
        self.lock = threading.Lock()

        # Базовые метрики пула/IPC
        self.register_counter("zerolink_pool_allocation_total", "Total number of pool allocations")
        self.register_gauge("zerolink_pool_used_bytes", "Current bytes used in pool")
        self.register_counter("zerolink_ipc_transmit_bytes_total", "Total bytes transmitted via IPC")
        self.register_gauge("zerolink_ipc_active_leases", "Number of active IPC leases")
        self.register_histogram("zerolink_pool_allocation_latency_seconds", "Pool allocation latency in seconds")
        self.register_gauge("zerolink_fragmentation_percentage", "Pool fragmentation percentage")

        # Новые метрики reliability/operations
        self.register_histogram("zerolink_runtime_operation_latency_seconds", "Latency of runtime/worker/server operations")
        self.register_counter("zerolink_alloc_failures_total", "Total number of allocation/import failures")
        self.register_counter("zerolink_lease_events_total", "Total number of lease state transitions")

    def register_metric(self, name: str, metric_type: MetricType, description: str = ""):
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = Metric(name=name, type=metric_type, description=description)

    def register_counter(self, name: str, description: str = ""):
        self.register_metric(name, MetricType.COUNTER, description)

    def register_gauge(self, name: str, description: str = ""):
        self.register_metric(name, MetricType.GAUGE, description)

    def register_histogram(self, name: str, description: str = ""):
        self.register_metric(name, MetricType.HISTOGRAM, description)

    def increment_counter(self, name: str, amount: float = 1.0, labels: Optional[Dict[str, str]] = None):
        with self.lock:
            if name in self.metrics:
                self.metrics[name].value += amount
                if labels:
                    self.metrics[name].labels.update(labels)

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        with self.lock:
            if name in self.metrics:
                self.metrics[name].value = value
                if labels:
                    self.metrics[name].labels.update(labels)

    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        # Упрощенная модель: храним последнее значение.
        with self.lock:
            if name in self.metrics:
                self.metrics[name].value = value
                if labels:
                    self.metrics[name].labels.update(labels)

    def get_metrics(self) -> Dict[str, Metric]:
        with self.lock:
            return self.metrics.copy()

    def get_prometheus_format(self) -> str:
        output = []
        for name, metric in self.get_metrics().items():
            if metric.description:
                output.append(f"# HELP {name} {metric.description}")
            output.append(f"# TYPE {name} {metric.type.value}")

            labels_str = ""
            if metric.labels:
                labels_parts = [f'{k}="{v}"' for k, v in metric.labels.items()]
                labels_str = "{" + ",".join(labels_parts) + "}"

            output.append(f"{name}{labels_str} {metric.value}")

        return "\n".join(output)


# Глобальный экземпляр телеметрии
telemetry = TelemetryCollector()


def monitor_ipc_transmission(bytes_count: int):
    """Мониторинг передачи данных через IPC."""
    telemetry.increment_counter("zerolink_ipc_transmit_bytes_total", bytes_count)


def update_fragmentation(fragmentation_percent: float):
    """Обновление метрики фрагментации."""
    telemetry.set_gauge("zerolink_fragmentation_percentage", fragmentation_percent)


def update_active_leases(count: int):
    """Обновление количества активных аренд."""
    telemetry.set_gauge("zerolink_ipc_active_leases", count)


def update_pool_usage(used_bytes: int):
    """Обновление использования пула."""
    telemetry.set_gauge("zerolink_pool_used_bytes", used_bytes)


def observe_runtime_latency(operation: str, seconds: float):
    """Наблюдение задержки runtime/worker/server операции."""
    telemetry.observe_histogram(
        "zerolink_runtime_operation_latency_seconds",
        seconds,
        labels={"operation": operation},
    )


def record_alloc_failure(component: str, reason: str = "unknown"):
    """Учет ошибок аллокации/импорта."""
    telemetry.increment_counter(
        "zerolink_alloc_failures_total",
        labels={"component": component, "reason": reason},
    )


def record_lease_event(component: str, event: str):
    """Учет событий lease churn (create/active/release/revoke...)."""
    telemetry.increment_counter(
        "zerolink_lease_events_total",
        labels={"component": component, "event": event},
    )


def monitor_pool_allocation(func: Callable) -> Callable:
    """Декоратор для мониторинга аллокаций пула."""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            telemetry.increment_counter("zerolink_pool_allocation_total")
            return result
        finally:
            latency = time.time() - start_time
            telemetry.observe_histogram("zerolink_pool_allocation_latency_seconds", latency)

    return wrapper
