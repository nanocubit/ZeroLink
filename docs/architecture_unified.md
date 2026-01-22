# Архитектура ZeroLink v2.0

## Общая архитектура

ZeroLink v2.0 представляет собой унифицированную систему для выполнения вычислений на GPU и CPU с поддержкой Zero-Copy IPC.

### Компоненты

1. **Runtime Layer**:
   - `ZeroLinkRuntime` - основной класс, объединяющий все компоненты
   - Поддерживает обратную совместимость через псевдоним `UnifiedRuntime`

2. **Core Layer**:
   - `Protocol` - система сообщений и сериализации
   - `GPU Subsystem` - управление GPU памятью через CUDA VMM
   - `CPU Subsystem` - управление CPU памятью через shared memory

3. **Server/Worker Layer**:
   - `MainIPCLeaseManager2P` - сервер для управления арендой памяти
   - `GPUWorker` - клиентская сторона для импорта GPU памяти

### Архитектурные принципы

- Zero-Copy IPC между процессами
- 2-Phase Lease Protocol для предотвращения Use-After-Free
- BLAKE3 для проверки целостности данных
- Поддержка CUDA Graphs и NVLink