from prometheus_client import Summary, Counter, start_http_server


class PipelineMetricsHandler:
    def __init__(self):
        self._summary_registry = {}
        self._counter_registry = {}

    def get_or_create_summary(self, name, description, labels=None):
        if name not in self._summary_registry:
            if labels:
                self._summary_registry[name] = Summary(name, description, labels)
            else:
                self._summary_registry[name] = Summary(name, description)
        return self._summary_registry[name]

    def get_or_create_counter(self, name, description, labels=None):
        if name not in self._counter_registry:
            if labels:
                self._counter_registry[name] = Counter(name, description, labels)
            else:
                self._counter_registry[name] = Counter(name, description)
        return self._counter_registry[name]


def start_monitoring_server(port: int = 8000):
    """Start the HTTP server to expose metrics."""
    return start_http_server(port)
