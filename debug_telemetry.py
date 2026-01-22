from zerolink.monitoring.telemetry import telemetry
telemetry.increment_counter('test_counter', 5)
telemetry.set_gauge('test_gauge', 42.0)
metrics = telemetry.get_metrics()
print('Keys:', list(metrics.keys()))
print('Values:', [type(v).__name__ for v in metrics.values()])
for k, v in metrics.items():
    print(f'{k}: {v.name if hasattr(v, "name") else v}')