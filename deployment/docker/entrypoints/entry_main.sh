#!/bin/bash
# Entry script для Main Server ZeroLink v2.0

echo "Starting ZeroLink Main Server..."

# Запуск main server
python -c "
from zerolink.runtime.unified import ZeroLinkRuntime
runtime = ZeroLinkRuntime()
runtime.start(block=True)
"