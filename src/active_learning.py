# src/active_learning.py
import numpy as np
from typing import List, Tuple
from src.hf_infer import predict, load_model
import math

# entropy_of_probs function computes the entropy of the probabilities.
# We use the formula -sum(p * log(p+eps) + (1-p) * log(1-p+eps) for p in probs) / len(probs).
# We return the entropy.
def entropy_of_probs(probs: List[float]) -> float:
    eps = 1e-12
    return -sum(p * math.log(p+eps) + (1-p) * math.log(1-p+eps) for p in probs) / len(probs)

# score_uncertainty function scores the uncertainty of the probabilities.
# We return the uncertainty score.
def score_uncertainty(probs: List[float]):
    # prefer high entropy
    return entropy_of_probs(probs)

# sample_uncertain function samples the uncertain texts.
# We load the model from the directory.
# We set the device to cuda if available, otherwise cpu.
# We tokenize the texts.
# We forward the inputs through the model.
# We compute the probabilities.
# We create a list of results.
# We iterate over the results and compute the uncertainty score.
# We return the uncertain texts.
def sample_uncertain(texts: List[str], model_dir: str, top_k: int = 50):
    tokenizer, model, inv_map = load_model(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    batch = 64
    results = []
    for i in range(0, len(texts), batch):
        batch_texts = texts[i:i+batch]
        res = predict(batch_texts, model_dir, device=device, threshold=0.0)
        for j, r in enumerate(res):
            probs = list(r["scores"].values())
            score = score_uncertainty(probs)
            results.append((score, batch_texts[j]))
    results.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in results[:top_k]]
