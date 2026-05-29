from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging
import sqlite3
import json
import sys
import os

# Ensure the root directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.aggregator import run_full_pipeline
from core.cache import set_cached

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Lie Detector API", description="Backend API for running the detection pipeline.")

# CORS — allow Next.js frontend from any origin in dev, restrict in prod
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SQLite path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SQLITE_DB_PATH = os.path.join(_ROOT, "results.db")


class AnalyzeRequest(BaseModel):
    query: str
    use_local_for_uncertainty: Optional[bool] = False


@app.post("/api/analyze")
def analyze_query(request: AnalyzeRequest):
    logger.info(f"Received API request for query: {request.query}")
    try:
        result_dict = run_full_pipeline(request.query, request.use_local_for_uncertainty)

        # Persist to SQLite for analytics dashboard
        set_cached(request.query, None, result_dict)

        # Convert HallucinationResult dataclass to dict so FastAPI can serialize it
        if "result" in result_dict and hasattr(result_dict["result"], "to_dict"):
            result_dict["result"] = result_dict["result"].to_dict()

        return result_dict
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
