# src/hf_infer.py
"""
Load trained HF model and predict multi-label classification for a list of texts.
Returns label scores (sigmoid on logits) and selected labels by a default threshold.
"""
import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# load_model function loads the model from the directory.
# We load the tokenizer and model from the directory.
# We set the problem type to multi-label classification.
# We load the label map from the JSON file.
# We create an inverse label map.
# We return the tokenizer, model, and inverse label map.
def load_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.config.problem_type = "multi_label_classification"
    # load label map
    with open(os.path.join(model_dir, "label_map.json"), "r", encoding="utf-8") as f:
        label_map = json.load(f)
    inv_label_map = {int(v):k for k,v in label_map.items()}
    return tokenizer, model, inv_label_map

# predict function predicts the labels for the given texts.
# We load the model from the directory.
# We set the device to cuda if available, otherwise cpu.
# We tokenize the texts.
# We forward the inputs through the model.
# We compute the probabilities.
# We create a list of results.
# We iterate over the probabilities and create a list of labels and scores.
# We return the results.
def predict(texts, model_dir, device=None, threshold=0.5):
    tokenizer, model, inv_label_map = load_model(model_dir)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()
    results = []
    for p in probs:
        labels = []
        scores = {}
        for idx, score in enumerate(p):
            lbl = inv_label_map.get(idx, str(idx))
            scores[lbl] = float(score)
            if score >= threshold:
                labels.append(lbl)
        results.append({"labels": labels, "scores": scores})
    return results
