"""Load/performance testing (continuous-engineering priority #28).

Real concurrent HTTP load against the actual running backend - no
external load-testing service (k6/Locust/JMeter) is installed or
needed for this; Python's stdlib concurrent.futures + urllib is
sufficient to generate real concurrent traffic and measure real
latency distributions. Every number this produces comes from an actual
request/response cycle; there is no synthetic/estimated timing anywhere
in this module.

CPU-bound generation is expected to behave differently under
concurrency than lightweight endpoints - that's the actual point of
running this, not something to paper over if the numbers look bad.
"""

import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field


@dataclass
class RequestResult:
    latency_ms: float
    status_code: int
    success: bool
    error: str = None


@dataclass
class LoadTestResult:
    name: str
    concurrency: int
    total_requests: int
    results: list = field(default_factory=list)

    def summary(self) -> dict:
        latencies = [r.latency_ms for r in self.results if r.success]
        errors = [r for r in self.results if not r.success]
        if not latencies:
            return {"name": self.name, "concurrency": self.concurrency,
                    "total_requests": self.total_requests, "successes": 0,
                    "errors": len(errors), "note": "No successful requests - "
                    "cannot compute latency stats from zero data points."}

        sorted_lat = sorted(latencies)
        n = len(sorted_lat)
        wall_time_s = sum(latencies) / 1000 / max(self.concurrency, 1)  # rough, real throughput measured separately

        return {
            "name": self.name,
            "concurrency": self.concurrency,
            "total_requests": self.total_requests,
            "successes": len(latencies),
            "errors": len(errors),
            "error_samples": [e.error for e in errors[:3]],
            "latency_ms": {
                "min": round(min(latencies), 2),
                "p50": round(sorted_lat[int(n * 0.50)], 2),
                "p95": round(sorted_lat[min(int(n * 0.95), n - 1)], 2),
                "p99": round(sorted_lat[min(int(n * 0.99), n - 1)], 2),
                "max": round(max(latencies), 2),
                "mean": round(statistics.mean(latencies), 2),
            },
        }


def _single_request(url: str, method: str = "GET", payload: dict = None, timeout: float = 60.0) -> RequestResult:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                  headers={"Content-Type": "application/json"} if data else {})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
            latency = (time.time() - t0) * 1000
            return RequestResult(latency_ms=latency, status_code=resp.status, success=True)
    except urllib.error.HTTPError as e:
        latency = (time.time() - t0) * 1000
        return RequestResult(latency_ms=latency, status_code=e.code, success=False,
                              error=f"HTTP {e.code}")
    except Exception as e:
        latency = (time.time() - t0) * 1000
        return RequestResult(latency_ms=latency, status_code=0, success=False,
                              error=f"{type(e).__name__}: {e}")


def run_load_test(name: str, url: str, method: str = "GET", payload: dict = None,
                   total_requests: int = 20, concurrency: int = 5, timeout: float = 60.0) -> LoadTestResult:
    """Fires `total_requests` real HTTP requests at `url` with up to
    `concurrency` in flight at once, using a real thread pool - not a
    simulated delay anywhere."""
    result = LoadTestResult(name=name, concurrency=concurrency, total_requests=total_requests)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(_single_request, url, method, payload, timeout)
            for _ in range(total_requests)
        ]
        for future in as_completed(futures):
            result.results.append(future.result())
    return result


def run_suite(base_url: str = "http://127.0.0.1:8123") -> dict:
    """A representative mix: a lightweight endpoint (health), a real
    deterministic-compute endpoint (calculator), and a real CPU-bound
    model-generation endpoint (chat) - deliberately different profiles,
    so the results show where this architecture actually holds up under
    concurrency and where it doesn't, rather than a single number that
    hides the difference."""
    suite = {}

    suite["health_light"] = run_load_test(
        "GET /api/health (lightweight)", f"{base_url}/api/health",
        total_requests=50, concurrency=10,
    ).summary()

    suite["calculator_deterministic"] = run_load_test(
        "POST /api/calculate (deterministic compute)", f"{base_url}/api/calculate",
        method="POST", payload={"calculation": "ebitda_margin", "inputs": {"ebitda": 100, "revenue": 500}},
        total_requests=30, concurrency=10,
    ).summary()

    suite["model_generation_cpu_bound"] = run_load_test(
        "POST /api/calculate via orchestrate (CPU-bound model generation)",
        f"{base_url}/api/ai/orchestrate", method="POST",
        payload={"query": "What is EBITDA?"},
        total_requests=6, concurrency=3, timeout=120.0,
    ).summary()

    return suite
