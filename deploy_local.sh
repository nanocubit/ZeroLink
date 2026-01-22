#!/bin/bash
# Скрипт локального деплоя ZeroLink v2.0

echo "=== Локальный деплой ZeroLink v2.0 ==="

# Установка зависимостей
echo "Установка зависимостей..."
pip install -r requirements.txt

# Сборка C++ расширений
echo "Сборка C++ расширений..."
python setup.py build_ext --inplace

# Проверка установки
echo "Проверка установки..."
python -c "from zerolink.runtime import ZeroLinkRuntime; print('ZeroLinkRuntime успешно импортирован')"

echo "Локальный деплой завершен"