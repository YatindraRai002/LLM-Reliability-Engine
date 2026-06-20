"""
Prometheus metrics collector module for the LLM Lie Detector API.
Defines metrics for tracking requests, latency, scores, risk labels, and cache hits.
"""
from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests processed",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "API request latency in seconds",
    ["endpoint"]
)

HALLUCINATION_SCORE = Histogram(
    "hallucination_score",
    "Distribution of hallucination scores returned by the pipeline",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

CACHE_HIT_COUNT = Counter(
    "cache_hits_total",
    "Total number of cache hits and misses",
    ["status"]
)

RISK_LABEL_COUNT = Counter(
    "risk_labels_total",
    "Total number of classifications per risk label",
    ["label"]
)

ACTIVE_REQUESTS = Gauge(
    "api_active_requests",
    "Number of active API requests currently running"
)
