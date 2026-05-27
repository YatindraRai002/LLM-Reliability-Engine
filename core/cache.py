import json
import hashlib
import sqlite3
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

# Try importing redis, gracefully fallback if not installed
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SQLITE_DB_PATH = os.path.join(CURRENT_DIR, "..", "results.db")

_redis_client = None

def get_redis_client():
    global _redis_client
    if not REDIS_AVAILABLE:
        return None
    if _redis_client is None:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        try:
            _redis_client = redis.from_url(redis_url, decode_responses=True)
            # Ping to verify connection
            _redis_client.ping()
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Falling back to SQLite only.")
            _redis_client = None
    return _redis_client


def _init_sqlite():
    """Initialize SQLite database for long-term storage."""
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                prompt TEXT,
                score REAL,
                label TEXT,
                cal REAL,
                unc REAL,
                cc REAL,
                weights TEXT
            )
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to initialize SQLite db: {e}")

# Initialize SQLite on module load
_init_sqlite()


def _prompt_key(prompt: str, weights: dict = None) -> str:
    """Generate a consistent cache key for a prompt and weight config."""
    w_str = json.dumps(weights, sort_keys=True) if weights else ""
    return hashlib.sha256(f"{prompt.strip()}_{w_str}".encode('utf-8')).hexdigest()


def get_cached(prompt: str, weights: dict = None) -> dict | None:
    """Check Redis for a cached result."""
    client = get_redis_client()
    if not client:
        return None

    key = _prompt_key(prompt, weights)
    try:
        cached_data = client.get(key)
        if cached_data:
            logger.info("Cache hit for prompt.")
            return json.loads(cached_data)
    except Exception as e:
        logger.warning(f"Redis get failed: {e}")
    return None


def set_cached(prompt: str, weights: dict = None, result_dict: dict = None, ttl: int = 3600):
    """Save to Redis (with TTL) and persist to SQLite."""
    if not result_dict:
        return

    key = _prompt_key(prompt, weights)

    # 1. Save to Redis
    client = get_redis_client()
    if client:
        try:
            client.setex(key, ttl, json.dumps(result_dict, default=str))
        except Exception as e:
            logger.warning(f"Redis set failed: {e}")

    # 2. Persist to SQLite
    _persist_to_sqlite(key, prompt, weights, result_dict)


def _persist_to_sqlite(key: str, prompt: str, weights: dict, result_dict: dict):
    """Save result metadata to SQLite for analytics."""
    try:
        # Extract fields from result_dict
        res = result_dict.get("result")
        if not res:
            return

        # Handle HallucinationResult dataclass or dict (if loaded from cache)
        if hasattr(res, "score"):
            score = res.score
            label = res.label
            cal = res.calibration_score
            unc = res.uncertainty_score
            cc = res.cross_check_score
        elif isinstance(res, dict):
            score = res.get("score")
            label = res.get("label")
            cal = res.get("calibration_score")
            unc = res.get("uncertainty_score")
            cc = res.get("cross_check_score")
        else:
            return

        weights_str = json.dumps(weights) if weights else "{}"
        timestamp = datetime.now().isoformat()

        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR IGNORE INTO results (id, timestamp, prompt, score, label, cal, unc, cc, weights)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (key, timestamp, prompt, score, label, cal, unc, cc, weights_str))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to persist to SQLite: {e}")
