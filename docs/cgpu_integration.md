# Интеграция с cgpu в ZeroLink v2.0

## Обзор

ZeroLink v2.0 предоставляет интеграцию с cgpu (https://github.com/nanocubit/cgpu) для улучшенного управления CUDA Driver API.

## Использование

```python
from zerolink.core.gpu import CgpuMemoryManager

# Использование cgpu для управления памятью
manager = CgpuMemoryManager()
```

## Особенности

- Совместимость с существующими CUDA контекстами
- Поддержка безопасного управления памятью
- Интеграция с системой аренды памяти ZeroLink