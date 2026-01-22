"""
ZeroLink - Unified Zero-Copy Runtime for CPU/GPU Computing
Copyright (C) 2025 Slava Maltsev <nanotec@live.ru>
SPDX-License-Identifier: MIT

zerolink/profiling/profiler.py

Система профилирования для ZeroLink v2.0.
"""

import torch
import time
import threading
from functools import wraps
from typing import Callable, Any, Dict, Optional


class Profiler:
    """Профилировщик для анализа производительности."""
    
    def __init__(self):
        self.profiles: Dict[str, Dict] = {}
        self.lock = threading.Lock()
        self.enabled = True
    
    def profile_function(self, name: str = None):
        """Декоратор для профилирования функции."""
        def decorator(func: Callable) -> Callable:
            nonlocal name
            if name is None:
                name = func.__name__
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                start_time = time.perf_counter()
                
                # Если это CUDA функция, используем torch.profiler
                if torch.cuda.is_available():
                    start_evt = torch.cuda.Event(enable_timing=True)
                    end_evt = torch.cuda.Event(enable_timing=True)
                    
                    start_evt.record()
                    result = func(*args, **kwargs)
                    end_evt.record()
                    
                    # Ждем завершения CUDA операций
                    torch.cuda.synchronize()
                    
                    cuda_time_ms = start_evt.elapsed_time(end_evt)
                else:
                    result = func(*args, **kwargs)
                    cuda_time_ms = 0.0
                
                end_time = time.perf_counter()
                
                total_time_ms = (end_time - start_time) * 1000
                
                with self.lock:
                    if name not in self.profiles:
                        self.profiles[name] = {
                            'call_count': 0,
                            'total_time_ms': 0.0,
                            'avg_time_ms': 0.0,
                            'cuda_time_ms': 0.0,
                            'max_time_ms': 0.0,
                            'min_time_ms': float('inf')
                        }
                    
                    profile = self.profiles[name]
                    profile['call_count'] += 1
                    profile['total_time_ms'] += total_time_ms
                    profile['cuda_time_ms'] += cuda_time_ms
                    profile['avg_time_ms'] = profile['total_time_ms'] / profile['call_count']
                    
                    if total_time_ms > profile['max_time_ms']:
                        profile['max_time_ms'] = total_time_ms
                    
                    if total_time_ms < profile['min_time_ms']:
                        profile['min_time_ms'] = total_time_ms
                
                return result
            return wrapper
        return decorator
    
    def get_profile_report(self) -> str:
        """Возвращает отчет о профилировании."""
        if not self.profiles:
            return "Нет данных профилирования"
        
        report_lines = ["Отчет профилирования:", "-" * 50]
        
        for name, data in sorted(self.profiles.items(), key=lambda x: x[1]['total_time_ms'], reverse=True):
            report_lines.append(f"{name}:")
            report_lines.append(f"  Вызовов: {data['call_count']}")
            report_lines.append(f"  Общее время: {data['total_time_ms']:.3f} ms")
            report_lines.append(f"  Среднее время: {data['avg_time_ms']:.3f} ms")
            report_lines.append(f"  Макс. время: {data['max_time_ms']:.3f} ms")
            report_lines.append(f"  Мин. время: {data['min_time_ms']:.3f} ms")
            report_lines.append(f"  CUDA время: {data['cuda_time_ms']:.3f} ms")
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def reset_profiles(self):
        """Сбросить все профили."""
        with self.lock:
            self.profiles.clear()
    
    def disable(self):
        """Отключить профилирование."""
        self.enabled = False
    
    def enable(self):
        """Включить профилирование."""
        self.enabled = True


# Глобальный профилировщик
profiler = Profiler()


def profile_gpu_function(name: str = None):
    """Декоратор для профилирования GPU функций."""
    return profiler.profile_function(name)