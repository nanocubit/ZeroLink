#!/bin/bash
# Скрипт деплоя в Docker для ZeroLink v2.0

echo "=== Деплой в Docker ZeroLink v2.0 ==="

# Проверка наличия Docker
command -v docker >/dev/null 2>&1 || { echo >&2 "Docker требуется, но не установлен. Выход."; exit 1; }

# Сборка Docker образа
echo "Сборка Docker образа..."
docker build -f deployment/docker/Dockerfile.unified -t zerolink:latest .

# Проверка сборки
if [ $? -eq 0 ]; then
    echo "Docker образ успешно собран"
    echo "Для запуска используйте: docker run -it zerolink:latest"
else
    echo "Ошибка при сборке Docker образа"
    exit 1
fi

echo "Деплой в Docker завершен"