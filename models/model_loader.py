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
def get_open_model():
    """Load the open-source LLM. Cached — loads only once per process."""
    model_cfg = CONFIG["models"]["local"]
    model_name = model_cfg["name"]
    use_4bit = model_cfg.get("use_4bit", False)

    logger.info(f"Loading open model: {model_name} (4bit: {use_4bit})")
    
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
            device_map="auto",
            torch_dtype=torch.float16 if not use_4bit else None,
            trust_remote_code=True
        )
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
