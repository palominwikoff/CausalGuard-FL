import torch
from torch import Tensor
from torch.nn import functional as F

from .model import AgreementModel


def invariance_penalty(logits: Tensor, labels: Tensor, silo: list[str]) -> Tensor:
    if logits.shape[0] != labels.shape[0] or len(silo) != labels.shape[0]:
        raise ValueError("silo labels must match the batch")
    penalty = logits.new_zeros(())
    for name in sorted(set(silo)):
        selection = torch.tensor([value == name for value in silo], device=logits.device)
        scale = torch.ones((), device=logits.device, dtype=logits.dtype, requires_grad=True)
        risk = F.cross_entropy(logits[selection] * scale, labels[selection])
        gradient = torch.autograd.grad(risk, scale, create_graph=True)[0]
        penalty = penalty + gradient.square()
    return penalty


def objective(model: AgreementModel, image: Tensor, text: Tensor, clinical: Tensor, labels: Tensor, silo: list[str], lambda_fd: float = 0.5, lambda_inv: float = 0.1) -> tuple[Tensor, dict[str, Tensor]]:
    logits, probabilities = model(image, text, clinical)
    task = F.cross_entropy(logits, labels)
    front_door = F.nll_loss(probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log(), labels)
    invariant = invariance_penalty(logits, labels, silo)
    total = task + lambda_fd * front_door + lambda_inv * invariant
    return total, {"task": task, "front_door": front_door, "invariance": invariant}
