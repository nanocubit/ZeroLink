# Решение проблемы `proxy 403 Forbidden` при установке зависимостей

## Почему возникает ошибка

Ошибка вида `Tunnel connection failed: 403 Forbidden` появляется не из-за `pip` или версии пакета, а из-за сетевой политики:

- окружение требует выход в интернет только через proxy (`HTTP_PROXY/HTTPS_PROXY`);
- proxy блокирует домены внешних Python-репозиториев (`pypi.org`, `files.pythonhosted.org`, `download.pytorch.org`).

Это подтверждается тем, что:

- с proxy запросы к `pypi.org` получают `403 Forbidden`;
- без proxy сеть недоступна (`[Errno 101] Network is unreachable`).

## Диагностика (что уже подтверждено в этом окружении)

- Через proxy запрос к `https://pypi.org/simple/numpy/` возвращает `403 Forbidden`.
- Без proxy прямой доступ в интернет недоступен (`Network is unreachable`).

Следовательно, единственный рабочий путь — предоставить доступ через разрешённый proxy к Python-репозиториям или использовать внутренний mirror/offline wheelhouse.

## Рабочие варианты решения

### Вариант A (рекомендуется): корпоративный PyPI mirror

Настроить внутренний mirror (Artifactory/Nexus/devpi) и использовать его как `index-url`.

```bash
pip config set global.index-url https://<internal-pypi>/simple
pip config set global.trusted-host <internal-pypi-host>
pip install -r requirements.txt
```

Для PyTorch (если mirror не кэширует его автоматически) — синхронизировать нужные wheels в mirror заранее.

### Вариант B: оффлайн wheelhouse

На машине с доступом в интернет:

```bash
pip download -r requirements.txt -d wheelhouse
```

Перенести `wheelhouse/` в изолированное окружение и установить локально:

```bash
pip install --no-index --find-links=wheelhouse -r requirements.txt
```

### Вариант C: разрешить proxy доступ к нужным хостам

Сетевой команде нужно разрешить CONNECT/HTTPS для:

- `pypi.org`
- `files.pythonhosted.org`
- `download.pytorch.org`

После этого стандартный `pip install -r requirements.txt` начинает работать без изменений в коде.

## Временный workaround для CI/локальной проверки

Если в окружении нельзя поставить `torch`, запускайте только синтаксические/документированные smoke-checks (например `python -m compileall -q ...` и IPC бенчмарк), а full `pytest` выполняйте в среде с предустановленным GPU-стеком.
