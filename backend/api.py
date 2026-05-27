from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging
import sys
import os

# Ensure the root directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.aggregator import run_full_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Lie Detector API", description="Backend API for running the detection pipeline.")

class AnalyzeRequest(BaseModel):
    query: str
    use_local_for_uncertainty: Optional[bool] = False

@app.post("/api/analyze")
async def analyze_query(request: AnalyzeRequest):
    logger.info(f"Received API request for query: {request.query}")
    try:
        # Run the synchronous pipeline (which handles its own event loop logic internally if needed,
        # or runs synchronously if we rewrote it to be fully synchronous)
        result_dict = run_full_pipeline(request.query, request.use_local_for_uncertainty)
        
        # Convert HallucinationResult dataclass to dict so FastAPI can serialize it
        if "result" in result_dict and hasattr(result_dict["result"], "to_dict"):
            result_dict["result"] = result_dict["result"].to_dict()
            
        return result_dict
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
