import torch
from torch import Tensor


def classification_metrics(probabilities: Tensor, labels: Tensor, bins: int = 10) -> dict[str, float]:
    if probabilities.ndim != 2 or labels.ndim != 1 or len(labels) != len(probabilities) or not len(labels):
        raise ValueError("invalid prediction and label shapes")
    if bins < 1 or not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("invalid probabilities or bin count")
    if not torch.allclose(probabilities.sum(dim=1), torch.ones(len(labels), device=probabilities.device, dtype=probabilities.dtype), atol=1e-5):
        raise ValueError("probability rows must sum to one")
    if (labels < 0).any() or (labels >= probabilities.shape[1]).any():
        raise ValueError("labels outside class range")
    confidence, prediction = probabilities.max(dim=1)
    correct = prediction.eq(labels)
    ece = probabilities.new_zeros(())
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = (confidence >= lower) & (confidence < upper if index < bins - 1 else confidence <= upper)
        if selected.any():
            ece = ece + selected.float().mean() * (confidence[selected].mean() - correct[selected].float().mean()).abs()
    return {"accuracy": correct.float().mean().item(), "ece": ece.item()}


def attack_success(predictions: Tensor, clean_labels: Tensor, target: int) -> float:
    if predictions.ndim != 1 or clean_labels.shape != predictions.shape:
        raise ValueError("predictions and labels must be matching vectors")
    eligible = clean_labels != target
    if not eligible.any():
        raise ValueError("attack success requires non-target examples")
    return predictions[eligible].eq(target).float().mean().item()
