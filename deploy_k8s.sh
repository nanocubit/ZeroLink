#!/bin/bash
# Скрипт деплоя в Kubernetes для ZeroLink v2.0

echo "=== Деплой в Kubernetes ZeroLink v2.0 ==="

# Проверка наличия kubectl
command -v kubectl >/dev/null 2>&1 || { echo >&2 "kubectl требуется, но не установлен. Выход."; exit 1; }

# Применение конфигурации
echo "Применение конфигурации..."
kubectl apply -f deployment/k8s/configmap.yaml
kubectl apply -f deployment/k8s/main_deployment.yaml
kubectl apply -f deployment/k8s/worker_deployment.yaml

# Проверка статуса
echo "Проверка статуса развёртывания..."
kubectl get deployments
kubectl get pods

echo "Деплой в Kubernetes завершен"
echo "Для просмотра логов используйте: kubectl logs -l app=zerolink-main"