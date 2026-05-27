import uuid
import time
import asyncio
import logging
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import AnalyzeRequest, JobStatus, HealthResponse
from core.aggregator import run_full_pipeline
from models.model_loader import get_system_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Lie Detector API",
    description="REST API for the hallucination detection pipeline.",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store (replace with Redis for production)
jobs = {}

START_TIME = time.time()

def process_job(job_id: str, request: AnalyzeRequest):
    """Background worker for processing the pipeline."""
    logger.info(f"Starting job {job_id}")
    try:
        # Currently, run_full_pipeline ignores the weights argument if not explicitly supported,
        # but the plan expects it to be supported in core.aggregator.
        # So we pass it as a kwarg. We need to update aggregator to accept it if it doesn't already.
        
        # NOTE: run_full_pipeline uses its own event loop to manage async tasks.
        # This is safe to run in a FastAPI thread pool.
        # To avoid passing unexpected kwargs if not yet updated:
        if request.weights:
            result = run_full_pipeline(request.prompt, weights=request.weights)
        else:
            result = run_full_pipeline(request.prompt)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = result
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@app.post("/analyze", response_model=JobStatus)
async def analyze_prompt(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """Submit a prompt for hallucination analysis."""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"job_id": job_id, "status": "pending", "result": None, "error": None}
    
    background_tasks.add_task(process_job, job_id, request)
    return JobStatus(**jobs[job_id])


@app.get("/result/{job_id}", response_model=JobStatus)
async def get_result(job_id: str):
    """Poll for the result of a submitted job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(**jobs[job_id])


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health and resource usage info."""
    uptime = time.time() - START_TIME
    sys_info = get_system_info()
    
    return HealthResponse(
        status="healthy",
        uptime_seconds=uptime,
        gpu_status=sys_info.get("gpu", "unknown"),
        models_loaded=sys_info.get("models_loaded", []),
        python_version=sys_info.get("python_version", "unknown"),
    )
