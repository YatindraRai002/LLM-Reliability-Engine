from fastapi import FastAPI, HTTPException, Response, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging
import sqlite3
import json
import sys
import os
import time
import uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.aggregator import run_full_pipeline
from core.cache import get_cached, set_cached
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from backend.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    HALLUCINATION_SCORE,
    CACHE_HIT_COUNT,
    RISK_LABEL_COUNT,
    ACTIVE_REQUESTS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Lie Detector API", description="Backend API for running the detection pipeline.")

@app.middleware("http")
async def add_metrics_middleware(request: Request, call_next):
    path = request.url.path
    if path == "/metrics":
        return await call_next(request)

    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    try:
        response = await call_next(request)
        status = str(response.status_code)
        REQUEST_COUNT.labels(method=request.method, endpoint=path, status=status).inc()
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint=path).observe(latency)
        return response
    except Exception as e:
        REQUEST_COUNT.labels(method=request.method, endpoint=path, status="500").inc()
        raise e
    finally:
        ACTIVE_REQUESTS.dec()

ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SQLITE_DB_PATH = os.path.join(_ROOT, "results.db")


class AnalyzeRequest(BaseModel):
    prompt: str
    use_local_for_uncertainty: Optional[bool] = False
    explain: Optional[bool] = True

jobs = {}


@app.get("/metrics")
def get_metrics():
    """Endpoint to expose Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def background_analyze(job_id: str, request: AnalyzeRequest):
    logger.info(f"Background processing started for job {job_id}, prompt: {request.prompt}")
    try:
        cached_result = get_cached(request.prompt)
        if cached_result:
            CACHE_HIT_COUNT.labels(status="hit").inc()
            res_val = cached_result.get("result")
            if res_val:
                score = res_val.get("score")
                label = res_val.get("label")
                if score is not None:
                    HALLUCINATION_SCORE.observe(score)
                if label is not None:
                    RISK_LABEL_COUNT.labels(label=label).inc()
            jobs[job_id] = {"status": "done", "result": cached_result}
            return

        CACHE_HIT_COUNT.labels(status="miss").inc()

        result_dict = run_full_pipeline(
            request.prompt,
            request.use_local_for_uncertainty,
            explain=request.explain if request.explain is not None else True,
        )

        set_cached(request.prompt, None, result_dict)

        res = result_dict.get("result")
        if res:
            score = getattr(res, "score", None)
            if score is None and isinstance(res, dict):
                score = res.get("score")
            
            label = getattr(res, "label", None)
            if label is None and isinstance(res, dict):
                label = res.get("label")

            if score is not None:
                HALLUCINATION_SCORE.observe(float(score))
            if label is not None:
                RISK_LABEL_COUNT.labels(label=label).inc()

        if "result" in result_dict and hasattr(result_dict["result"], "to_dict"):
            result_dict["result"] = result_dict["result"].to_dict()

        jobs[job_id] = {"status": "done", "result": result_dict}
    except Exception as e:
        logger.error(f"Pipeline error for job {job_id}: {e}")
        jobs[job_id] = {"status": "error", "error": str(e)}

@app.post("/analyze", status_code=202)
def analyze_query(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending"}
    background_tasks.add_task(background_analyze, job_id, request)
    return {"job_id": job_id, "status": "pending"}

@app.get("/result/{job_id}")
def get_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/api/history")
def get_history():
    """Return all analysis history from SQLite for the analytics dashboard."""
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT id, timestamp, prompt, score, label, cal, unc, cc, weights "
            "FROM results ORDER BY timestamp DESC LIMIT 200"
        )
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return {"history": rows, "total": len(rows)}
    except Exception as e:
        logger.error(f"Failed to load history: {e}")
        return {"history": [], "total": 0, "error": str(e)}


@app.delete("/api/history")
def clear_history():
    """Clear all analysis history from SQLite."""
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        conn.execute("DELETE FROM results")
        conn.commit()
        conn.close()
        return {"status": "cleared"}
    except Exception as e:
        logger.error(f"Failed to clear history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/system-calibration")
def get_system_calibration():
    """Return evaluation results for the system calibration tab."""
    eval_path = os.path.join(_ROOT, "eval_results.json")
    if not os.path.exists(eval_path):
        return {"available": False, "data": []}
    try:
        with open(eval_path, "r") as f:
            results = json.load(f)
        labeled = [r for r in results if r.get("correctness") is not None]
        return {"available": True, "data": labeled}
    except Exception as e:
        logger.error(f"Failed to load eval results: {e}")
        return {"available": False, "data": [], "error": str(e)}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
