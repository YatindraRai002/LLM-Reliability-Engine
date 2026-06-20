"""
models/groq_client.py
Groq API wrapper with proper error handling, retry logic,
parallel sampling, and graceful degradation flags.
"""

import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import yaml
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(_ROOT, "config.yaml")) as f:
    CONFIG = yaml.safe_load(f)

_groq_client = None


def _get_client():
    global _groq_client
    if _groq_client is None:
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("Run: pip install groq")
        
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        
        # Validate key format before even trying
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set in .env file.\n"
                "Get a free key at https://console.groq.com/keys"
            )
        if not api_key.startswith("gsk_"):
            raise EnvironmentError(
                f"GROQ_API_KEY looks wrong (got: {api_key[:8]}...). "
                "Valid Groq keys start with 'gsk_'."
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def groq_generate(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    system: str = "You are a helpful, accurate assistant. Answer concisely and factually.",
    retries: int = 3,
) -> str:
    """Single response from Groq with exponential backoff retry."""
    client = _get_client()
    model  = CONFIG["models"]["groq"]["name"]
    
    # Truncate prompt to avoid token overflow
    if len(prompt) > 4000:
        prompt = prompt[:3500] + "\n...[truncated]...\n" + prompt[-500:]
    
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            err_str = str(e)
            
            # 401 = invalid key — no point retrying
            if "401" in err_str or "Invalid API Key" in err_str:
                raise EnvironmentError(
                    f"Groq API key is invalid (401). "
                    f"Go to https://console.groq.com/keys and create a new key.\n"
                    f"Raw error: {err_str}"
                )
            
            # 429 = rate limit — wait and retry
            if "429" in err_str or "rate" in err_str.lower():
                wait = 2.0 * (2 ** attempt)  # 2s, 4s, 8s
                logger.warning(f"Groq rate limit — waiting {wait}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
                continue
            
            # Other transient errors
            if attempt < retries - 1:
                time.sleep(2.0)
                continue
            
            raise  # re-raise on final attempt


def groq_generate_n_parallel(
    prompt: str,
    n: int = None,
    temperature: float = None,
) -> list:
    """Generate N responses in parallel. Returns list of strings."""
    n    = n    or CONFIG["sampling"]["n_samples"]
    temp = temperature or CONFIG["sampling"]["temperature"]
    
    stagger_ms = CONFIG["sampling"].get("request_stagger_ms", 300)
    max_workers = CONFIG["sampling"].get("max_concurrent_requests", 3)
    
    logger.info(f"Requesting {n} Groq samples at temp={temp} with stagger {stagger_ms}ms and {max_workers} concurrent workers")
    
    def _call(i: int) -> str:
        time.sleep(i * (stagger_ms / 1000.0))  # stagger
        return groq_generate(prompt, temperature=temp)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_call, i) for i in range(n)]
        for f in as_completed(futures):
            try:
                text = f.result()
                if text.strip():
                    results.append(text)
                    logger.info(f"  Got sample {len(results)}/{n}")
            except Exception as e:
                logger.error(f"  Sample failed: {e}")

    logger.info(f"Sampling complete: {len(results)}/{n} succeeded")
    
    if len(results) < 2:
        raise RuntimeError(
            f"Too many Groq failures: only {len(results)}/{n} samples. "
            "Need at least 2. Check your API key and rate limits."
        )
    return results



def groq_cross_check(prompt: str) -> str:
    """Deterministic Groq response for cross-checking."""
    return groq_generate(
        prompt,
        temperature=0.0,
        system=(
            "You are a precise, factual assistant. "
            "Answer directly and concisely. "
            "If uncertain, say so explicitly."
        ),
    )


def safe_groq_cross_check(prompt: str) -> dict:
    """
    Cross-check wrapper that NEVER raises — returns a result dict
    with groq_available=False on any failure so the aggregator
    can degrade gracefully instead of crashing or giving wrong scores.
    """
    try:
        response = groq_cross_check(prompt)
        return {
            "groq_response":   response,
            "groq_available":  True,
            "error":           None,
        }
    except EnvironmentError as e:
        # Invalid key — user needs to fix this
        logger.error(f"Groq key error: {e}")
        return {
            "groq_response":   None,
            "groq_available":  False,
            "error":           str(e),
            "error_type":      "invalid_key",
        }
    except Exception as e:
        logger.error(f"Groq cross-check failed: {e}")
        return {
            "groq_response":   None,
            "groq_available":  False,
            "error":           str(e),
            "error_type":      "api_error",
        }


def test_groq_connection() -> dict:
    """
    Test if Groq is reachable with current key.
    Returns dict with success bool and error message if failed.
    """
    try:
        reply = groq_generate(
            "Reply with exactly these three words: CONNECTION_TEST_OK",
            max_tokens=15,
            retries=1,
        )
        success = "CONNECTION_TEST_OK" in reply
        return {"success": success, "response": reply, "error": None}
    except Exception as e:
        return {"success": False, "response": None, "error": str(e)}
