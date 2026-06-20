"""
Prometheus metrics collector module for the LLM Lie Detector API.
Defines metrics for tracking requests, latency, scores, risk labels, and cache hits.
"""
from prometheus_client import Counter, Histogram, Gauge

# Track total API requests
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests processed",
    ["method", "endpoint", "status"]
)

# Track request duration/latency
REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "API request latency in seconds",
    ["endpoint"]
)

# Track distribution of hallucination scores
HALLUCINATION_SCORE = Histogram(
    "hallucination_score",
    "Distribution of hallucination scores returned by the pipeline",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Track cache hit and miss counts
CACHE_HIT_COUNT = Counter(
    "cache_hits_total",
    "Total number of cache hits and misses",
    ["status"]  # "hit" or "miss"
)

# Track risk label counts (low, medium, high)
RISK_LABEL_COUNT = Counter(
    "risk_labels_total",
    "Total number of classifications per risk label",
    ["label"]  # "low", "medium", "high"
)

# Track active requests
ACTIVE_REQUESTS = Gauge(
    "api_active_requests",
    "Number of active API requests currently running"
)
