import torch
from torch import Tensor


def projected_average(updates: Tensor, weights: Tensor, coordinate_mask: Tensor, rank: int) -> tuple[Tensor, Tensor]:
    if updates.ndim != 2 or updates.shape[0] < 2 or updates.shape[1] < 1:
        raise ValueError("updates must contain at least two client vectors")
    clients, parameters = updates.shape
    if weights.shape != (clients,) or coordinate_mask.shape != (parameters,):
        raise ValueError("weight or mask shape mismatch")
    if coordinate_mask.dtype != torch.bool:
        raise ValueError("coordinate mask must be boolean")
    if not torch.isfinite(updates).all() or not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("updates and weights must be finite and weights nonnegative")
    weights = weights / weights.sum()
    mean = (weights[:, None] * updates).sum(dim=0)
    centered = (updates - mean).T * coordinate_mask[:, None]
    u, singular_values, _ = torch.linalg.svd(centered, full_matrices=False)
    maximum = min(int(coordinate_mask.sum().item()), int(torch.linalg.matrix_rank(centered).item()))
    if rank < 0 or rank > maximum:
        raise ValueError("rank exceeds disagreement rank")
    basis = u[:, :rank]
    projected = mean - basis @ (basis.T @ mean)
    return projected, singular_values
