from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSequenceClassification
import torch
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import yaml
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FIX: Find config.yaml relative to this file's directory (models/)
# The config.yaml is one level up from the 'models' folder.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(CURRENT_DIR, "..", "config.yaml")

def load_config():
    if not os.path.exists(CONFIG_PATH):
        logger.error(f"Config file {CONFIG_PATH} not found!")
        raise FileNotFoundError(f"Missing {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

CONFIG = load_config()

@lru_cache(maxsize=1)
def get_local_model():
    """Load the open-source LLM. Cached — loads only once per process."""
    model_cfg = CONFIG["models"]["local"]
    model_name = model_cfg["name"]
    use_4bit = model_cfg.get("use_4bit", False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning("CUDA not found. Disabling 4-bit quantization and using float32 for CPU performance.")
        use_4bit = False
        dtype = torch.float32
    else:
        dtype = torch.float16 if not use_4bit else None

    logger.info(f"Loading open model: {model_name} (4bit: {use_4bit}, dtype: {dtype})")
    
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto" if use_4bit else None,
            dtype=dtype,               # torch_dtype renamed to dtype in transformers 5.x
            trust_remote_code=True
        )
        if not use_4bit:
            model = model.to(device)
        model.eval()
        return tokenizer, model
    except Exception as e:
        logger.error(f"Failed to load open model: {e}")
        raise

@lru_cache(maxsize=1)
def get_embedding_model():
    model_name = CONFIG["models"]["embedding"]["name"]
    logger.info(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)

@lru_cache(maxsize=1)
def get_nli_model():
    model_name = CONFIG["models"]["nli"]["name"]
    logger.info(f"Loading NLI model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

def get_system_info() -> dict:
    """Returns system hardware and loaded model status."""
    info = {
        "python_version": sys.version.split(" ")[0],
        "models_loaded": [],
    }

    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        vram_alloc = torch.cuda.memory_allocated(0) / (1024**3)
        vram_res = torch.cuda.memory_reserved(0) / (1024**3)
        info["vram_usage"] = f"{vram_alloc:.2f}GB alloc / {vram_res:.2f}GB res"
    else:
        info["gpu"] = "CPU-only"
        info["vram_usage"] = "N/A"

    # Check which models are loaded in lru_cache
    if get_local_model.cache_info().currsize > 0:
        info["models_loaded"].append(CONFIG["models"]["local"]["name"])
    if get_embedding_model.cache_info().currsize > 0:
        info["models_loaded"].append(CONFIG["models"]["embedding"]["name"])
    if get_nli_model.cache_info().currsize > 0:
        info["models_loaded"].append(CONFIG["models"]["nli"]["name"])

    return info

# Deprecated alias, remove after full audit
get_open_model = get_local_model
