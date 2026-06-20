import asyncio
import numpy as np
import torch
import yaml
import logging
from sklearn.cluster import AgglomerativeClustering
from models.model_loader import get_local_model, get_embedding_model, CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Removed the global gpu_lock = asyncio.Lock() from here

def _format_prompt(user_msg: str, model_name: str) -> str:
    """Apply the correct chat template per model."""
    if "tinyllama" in model_name.lower():
        return (
            "<|system|>\nYou are a helpful assistant.</s>\n"
            f"<|user|>\n{user_msg.strip()}</s>\n"
            "<|assistant|>\n"
        )
    elif "mistral" in model_name.lower():
        return f"<s>[INST] {user_msg.strip()} [/INST]"
    else:
        return user_msg

async def generate_n_samples_batch(
    prompt: str,
    n: int = None,
    temperature: float = None
) -> list[str]:
    """
    Generate N diverse responses using a single batch call to the GPU.
    Optimized with KV caching and performance settings.
    """
    n = n or CONFIG["sampling"].get("n_samples", 3)
    temp = temperature or CONFIG["sampling"].get("temperature", 0.8)
    max_tokens = 200

    logger.info(f"Generating {n} samples with temperature {temp} in a single batch...")

    lock = asyncio.Lock()
    async with lock:
        try:
            tokenizer, model = get_local_model()
            model_name = CONFIG["models"]["local"]["name"]
            formatted = _format_prompt(prompt, model_name)
            inputs = tokenizer([formatted] * n, return_tensors="pt", padding=True).to(model.device)
            input_length = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    min_new_tokens=30,
                    do_sample=True,
                    temperature=temp,
                    top_p=0.95,
                    top_k=50, # Limit search space for speed
                    use_cache=True, # CRITICAL: Enable KV Caching
                    repetition_penalty=1.15,
                    pad_token_id=tokenizer.eos_token_id
                )

            responses = []
            for i in range(n):
                generated = outputs[i][input_length:]
                responses.append(tokenizer.decode(generated, skip_special_tokens=True).strip())

            return responses
        except Exception as e:
            logger.error(f"Batch generation error: {e}")
            return [""] * n

def _project_2d(embeddings: np.ndarray) -> np.ndarray:
    n = embeddings.shape[0]
    
    if n < 2:
        return np.zeros((n, 2), dtype=np.float32)
    
    if n == 2:
        # PCA with 2 samples gives degenerate dim 2
        # Just place them on a line with unit spacing
        vec = embeddings[1] - embeddings[0]
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            # Identical embeddings — place side by side
            return np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        unit = vec / norm
        proj0 = float(np.dot(embeddings[0], unit))
        proj1 = float(np.dot(embeddings[1], unit))
        return np.array([[proj0, 0.0], [proj1, 0.0]], dtype=np.float32)
    
    # Normal PCA for n >= 3
    from sklearn.decomposition import PCA
    
    # Check for degenerate case: near-zero variance
    variance = np.var(embeddings, axis=0).sum()
    if variance < 1e-8:
        np.random.seed(42)
        jitter = np.random.normal(0, 0.01, embeddings.shape)
        embeddings = embeddings + jitter
    
    pca    = PCA(n_components=2)
    coords = pca.fit_transform(embeddings).astype(np.float32)
    return coords

def compute_semantic_uncertainty(responses: list[str]) -> dict:
    """
    Compute semantic uncertainty using embedding clustering.
    """
    if not responses or all(not r for r in responses):
        return {"uncertainty_score": 1.0, "n_semantic_clusters": 1, "responses": responses}

    embedder = get_embedding_model()

    embeddings = embedder.encode(responses, convert_to_tensor=True, normalize_embeddings=True)
    sim_matrix = (embeddings @ embeddings.T).cpu().numpy()

    distance_matrix = 1.0 - sim_matrix
    np.fill_diagonal(distance_matrix, 0)

    # Fixed: Config no longer has similarity_threshold, use a sensible default (0.7)
    threshold = 0.7

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1.0 - threshold,
        metric="precomputed",
        linkage="average"
    )

    labels = clustering.fit_predict(distance_matrix)
    n_clusters = len(set(labels))

    n = len(responses)
    mask = ~np.eye(n, dtype=bool)
    mean_similarity = float(sim_matrix[mask].mean()) if n > 1 else 1.0
    uncertainty_score = float(1.0 - mean_similarity)

    cluster_counts = np.bincount(labels)
    cluster_probs = cluster_counts / cluster_counts.sum()
    semantic_entropy = float(-np.sum(cluster_probs * np.log(cluster_probs + 1e-10)))

    max_entropy = np.log(n) if n > 1 else 1.0
    normalized_entropy = semantic_entropy / max_entropy if max_entropy > 0 else 0.0

    # Project to 2D for the scatter plot
    embeddings_np = embeddings.cpu().numpy()
    coords_2d = _project_2d(embeddings_np).tolist()

    result = {
        "uncertainty_score": uncertainty_score,
        "normalized_entropy": normalized_entropy,
        "n_semantic_clusters": n_clusters,
        "mean_pairwise_similarity": mean_similarity,
        "cluster_labels": labels.tolist(),
        "embeddings_2d": coords_2d,
        "responses": responses
    }
    logger.info(
        f"Uncertainty={result['uncertainty_score']:.3f} "
        f"clusters={result['n_semantic_clusters']} "
        f"keys={list(result.keys())}"
    )
    return result

def run_semantic_uncertainty_pipeline(prompt: str, use_local: bool = False) -> dict:
    import asyncio
    responses = asyncio.run(generate_n_samples_batch(prompt))
    return compute_semantic_uncertainty(responses)

async def run_semantic_uncertainty_pipeline_async(prompt: str) -> dict:
    """Async pipeline orchestrator."""
    responses = await generate_n_samples_batch(prompt)
    return compute_semantic_uncertainty(responses)
