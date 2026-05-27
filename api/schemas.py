from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class AnalyzeRequest(BaseModel):
    prompt: str = Field(..., max_length=10000, description="The user query to analyze.")
    weights: Optional[Dict[str, float]] = Field(None, description="Optional custom weights for the signals.")

class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending", "completed", "failed"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    gpu_status: str
    models_loaded: list[str]
    python_version: str
