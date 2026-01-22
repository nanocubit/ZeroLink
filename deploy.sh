#!/bin/bash
# Скрипт автоматизации деплоя ZeroLink v2.0

set -e  # Выход при ошибке

echo "=== Скрипт автоматизации деплоя ZeroLink v2.0 ==="

# Проверка наличия необходимых инструментов
command -v python3 >/dev/null 2>&1 || { echo >&2 "Python3 требуется, но не установлен. Выход."; exit 1; }
command -v pip >/dev/null 2>&1 || { echo >&2 "Pip требуется, но не установлен. Выход."; exit 1; }
command -v git >/dev/null 2>&1 || { echo >&2 "Git требуется, но не установлен. Выход."; exit 1; }

# Параметры по умолчанию
ACTION=${1:-"deploy"}
TARGET=${2:-"local"}

echo "Действие: $ACTION"
echo "Цель: $TARGET"

case $ACTION in
    "install")
        echo "Установка ZeroLink v2.0..."
        pip install -r requirements.txt
        python setup.py build_ext --inplace
        echo "ZeroLink v2.0 успешно установлен"
        ;;
    "test")
        echo "Запуск тестов..."
        python -m pytest tests/ -v
        python test_basic_functionality.py
        python test_basic_functionality_fixed.py
        echo "Тесты завершены"
        ;;
    "demo")
        echo "Запуск демо-скрипта..."
        python demo.py
        echo "Демо-скрипт завершен"
        ;;
    "deploy")
        case $TARGET in
            "local")
                echo "Локальный деплой ZeroLink v2.0..."
                ./deploy_local.sh
                ;;
            "docker")
                echo "Деплой в Docker..."
                ./deploy_docker.sh
                ;;
            "k8s")
                echo "Деплой в Kubernetes..."
                ./deploy_k8s.sh
                ;;
            *)
                echo "Неизвестная цель деплоя: $TARGET"
                echo "Доступные цели: local, docker, k8s"
                exit 1
                ;;
        esac
        ;;
    "build-docker")
        echo "Сборка Docker образа..."
        docker build -f deployment/docker/Dockerfile.unified -t zerolink:latest .
        echo "Docker образ собран"
        ;;
    *)
        echo "Неизвестное действие: $ACTION"
        echo "Доступные действия: install, test, demo, deploy, build-docker"
        exit 1
        ;;
esac

echo "=== Деплой ZeroLink v2.0 завершен ==="