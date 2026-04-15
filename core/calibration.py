import torch
import numpy as np
from models.model_loader import get_open_model
import logging

logger = logging.getLogger(__name__)

def get_generation_with_scores(prompt: str, max_new_tokens: int = 64) -> dict:
    try:
        tokenizer, model = get_open_model()

        # TinyLlama Chat Format
        formatted = f"<|system|>\nYou are a helpful, honest, and factual assistant. If you are unsure about a fact or acronym, state that you do not know rather than guessing.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            )

        generated_ids = outputs.sequences[0][input_length:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        if len(generated_ids) == 0:
            return {"response": "", "token_probs": [], "mean_confidence": 0.0, "min_confidence": 0.0, "confidence_std": 0.0}

        token_probs = []
        for i, token_id in enumerate(generated_ids):
            score_tensor = outputs.scores[i]
            probs = torch.softmax(score_tensor[0], dim=-1)
            chosen_token_prob = probs[token_id].item()
            token_probs.append(chosen_token_prob)

        return {
            "response": response_text,
            "token_probs": token_probs,
            "mean_confidence": float(np.mean(token_probs)),
            "min_confidence": float(np.min(token_probs)),
            "confidence_std": float(np.std(token_probs))
        }
    except Exception as e:
        logger.error(f"Error in get_generation_with_scores: {e}")
        raise

def compute_calibration_score(token_probs: list) -> float:
    if not token_probs: return 1.0
    probs = np.array(token_probs)
    mean_conf = np.mean(probs)
    std_conf = np.std(probs)
    return float(np.clip(0.8 * (1.0 - mean_conf) + 0.2 * min(std_conf / 0.2, 1.0), 0.0, 1.0))

def compute_calibration_from_batch(responses: list[str]) -> float:
    """
    Estimate calibration score from a batch of diverse responses.
    Since we don't have logits for the batch generation, we use semantic
    consistency as a proxy for calibration.
    """
    if not responses or all(not r for r in responses):
        return 1.0

    # If responses are highly diverse, the model is likely uncertain (high score)
    # We reuse the logic from semantic uncertainty but simplify it for a single score.
    # In a production system, we would capture logits during batch_generate.

    # For now, we use 1.0 - mean_similarity as a robust proxy for calibration
    # when we only have text outputs.
    try:
        from models.model_loader import get_embedding_model
        embedder = get_embedding_model()
        embeddings = embedder.encode(responses, convert_to_tensor=True, normalize_embeddings=True)
        sim_matrix = (embeddings @ embeddings.T).cpu().numpy()
        mask = ~np.eye(len(responses), dtype=bool)
        mean_similarity = float(sim_matrix[mask].mean()) if len(responses) > 1 else 1.0
        return float(np.clip(1.0 - mean_similarity, 0.0, 1.0))
    except Exception as e:
        logger.error(f"Error computing batch calibration: {e}")
        return 0.5

def compute_ece(confidence_bins: list, accuracy_bins: list, n_bins: int = 10) -> float:
    if not confidence_bins or not accuracy_bins: return 0.0
    confidences, accuracies = np.array(confidence_bins), np.array(accuracy_bins)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i+1])
        if mask.sum() == 0: continue
        ece += mask.mean() * abs(accuracies[mask].mean() - confidences[mask].mean())
    return float(ece)
