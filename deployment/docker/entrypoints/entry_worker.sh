#!/bin/bash
# Entry script для Worker Server ZeroLink v2.0

echo "Starting ZeroLink Worker Server..."

# Запуск worker server
python -c "
from zerolink.workers.gpu_worker import GPUWorker
worker = GPUWorker(sock_path='/tmp/zerolink.sock', device_id=0)
worker.connect()

while True:
    worker.loop_once()
"