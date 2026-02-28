from prometheus_client import Counter, Gauge, Histogram

request_count = Counter(
    "request_count_total",
    "Total HTTP requests received",
    ["method", "endpoint", "status_code"],
)

inference_latency_seconds = Histogram(
    "inference_latency_seconds",
    "Inference latency in seconds",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1, 2, 5),
)

active_requests = Gauge(
    "active_requests",
    "Number of in-flight HTTP requests",
)

triton_errors_total = Counter(
    "triton_errors_total",
    "Total number of Triton inference or connectivity errors",
)
