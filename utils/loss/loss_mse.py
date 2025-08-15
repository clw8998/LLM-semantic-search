from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from sentence_transformers import SentenceTransformer


class MSELoss(nn.Module):
    def __init__(self, model: SentenceTransformer, scale: float = 1.0) -> None:
        super().__init__()
        self.model = model
        self.loss_fct = nn.MSELoss()
        self.scale = scale

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        # Concatenate multiple inputs on the batch dimension
        if len(sentence_features) > 1:
            embeddings = torch.cat([self.model(inputs)["sentence_embedding"] for inputs in sentence_features], dim=0)
            # Repeat the labels for each input
            return self.loss_fct(embeddings, labels.repeat(len(sentence_features), 1))

        embeddings = self.model(sentence_features[0])["sentence_embedding"]

        emb_dim = embeddings.size(1)
        if labels.size(1) > emb_dim:
            labels = labels[:, :emb_dim] 

        return self.loss_fct(embeddings, labels) * self.scale
