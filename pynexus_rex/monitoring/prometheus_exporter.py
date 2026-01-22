"""
zerolink/monitoring/prometheus_exporter.py

Экспорт метрик в формате Prometheus.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
from ..monitoring.telemetry import telemetry


class PrometheusMetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            
            metrics_text = telemetry.get_prometheus_format()
            self.wfile.write(metrics_text.encode('utf-8'))
        elif self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = """
            <html>
            <head><title>ZeroLink Metrics</title></head>
            <body>
            <h1>ZeroLink v2.0 Metrics</h1>
            <p><a href='/metrics'>Metrics</a></p>
            </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()


class PrometheusExporter:
    """
    Экспортер метрик Prometheus.
    """
    def __init__(self, host='localhost', port=8000):
        self.host = host
        self.port = port
        self.httpd = None
        self.server_thread = None
    
    def start(self):
        """Запускает HTTP сервер для экспорта метрик."""
        self.httpd = HTTPServer((self.host, self.port), PrometheusMetricsHandler)
        self.server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.server_thread.start()
        print(f"[PrometheusExporter] Started on http://{self.host}:{self.port}/metrics")
    
    def stop(self):
        """Останавливает HTTP сервер."""
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
        
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=2.0)
        
        print("[PrometheusExporter] Stopped")


# Глобальный экземпляр экспортера
prometheus_exporter = PrometheusExporter()