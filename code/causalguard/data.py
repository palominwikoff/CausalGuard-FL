import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset


class FeatureDataset(Dataset[dict[str, Any]]):
    def __init__(self, path: str | Path):
        self.rows = []
        with Path(path).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            required = {"patient", "silo", "image_features", "text_features", "clinical_features", "label"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError("CSV is missing required fields")
            for row in reader:
                image = self._vector(row["image_features"])
                text = self._vector(row["text_features"])
                clinical = self._vector(row["clinical_features"])
                if image.shape != text.shape:
                    raise ValueError("image and text embeddings must have the same shape")
                label = int(row["label"])
                if label < 0:
                    raise ValueError("label must be nonnegative")
                self.rows.append({"patient": row["patient"], "silo": row["silo"], "image": image, "text": text, "clinical": clinical, "label": label})
        if not self.rows:
            raise ValueError("dataset has no rows")
        shape = tuple(self.rows[0][name].shape for name in ("image", "text", "clinical"))
        for row in self.rows:
            if tuple(row[name].shape for name in ("image", "text", "clinical")) != shape:
                raise ValueError("feature dimensions differ across rows")
            if not row["patient"] or not row["silo"]:
                raise ValueError("patient and silo must be nonempty")

    @staticmethod
    def _vector(value: str) -> Tensor:
        values = json.loads(value)
        if not isinstance(values, list) or not values:
            raise ValueError("each feature must be a nonempty JSON array")
        tensor = torch.tensor(values, dtype=torch.float32)
        if tensor.ndim != 1 or not torch.isfinite(tensor).all():
            raise ValueError("features must be finite one-dimensional arrays")
        return tensor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def check_patient_disjoint(*datasets: FeatureDataset) -> None:
    seen: set[str] = set()
    for dataset in datasets:
        current = {row["patient"] for row in dataset.rows}
        if seen.intersection(current):
            raise ValueError("patient overlap between datasets")
        seen.update(current)
