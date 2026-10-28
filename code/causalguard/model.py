import torch
from torch import Tensor, nn
from torch.nn import functional as F


def agreement_embedding(image: Tensor, text: Tensor) -> Tensor:
    if image.shape != text.shape or image.ndim != 2:
        raise ValueError("paired embeddings must have matching batch and feature dimensions")
    cosine = F.cosine_similarity(image, text, dim=-1).unsqueeze(-1)
    return torch.cat((cosine, image * text), dim=-1)


class AgreementModel(nn.Module):
    def __init__(self, feature_dim: int, clinical_dim: int, classes: int, centroids: Tensor, hidden_dim: int = 32, temperature: float = 1.0):
        super().__init__()
        if feature_dim < 1 or clinical_dim < 1 or classes < 2 or hidden_dim < 1:
            raise ValueError("invalid model dimensions")
        if centroids.ndim != 2 or centroids.shape[1] != feature_dim + 1 or not torch.isfinite(centroids).all():
            raise ValueError("centroids must match the agreement dimension")
        if centroids.shape[0] < 1 or temperature <= 0:
            raise ValueError("centroids and temperature must be positive")
        self.register_buffer("centroids", centroids.detach().clone().float())
        self.temperature = temperature
        input_dim = feature_dim * 2 + clinical_dim
        self.task = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, classes))
        self.outcome = nn.Sequential(nn.Linear(input_dim + feature_dim + 1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, classes))

    def assignment(self, image: Tensor, text: Tensor) -> Tensor:
        mediator = agreement_embedding(image, text)
        distance = torch.cdist(mediator, self.centroids).square()
        return torch.softmax(-distance / self.temperature, dim=-1)

    def forward(self, image: Tensor, text: Tensor, clinical: Tensor) -> tuple[Tensor, Tensor]:
        inputs = torch.cat((image, text, clinical), dim=-1)
        assignment = self.assignment(image, text)
        count = self.centroids.shape[0]
        batch = inputs.shape[0]
        m = self.centroids[:, None, :].expand(count, batch, -1)
        x = inputs[None, :, :].expand(count, batch, -1)
        outcomes = torch.softmax(self.outcome(torch.cat((m, x), dim=-1)), dim=-1).mean(dim=1)
        estimate = assignment @ outcomes
        return self.task(inputs), estimate
