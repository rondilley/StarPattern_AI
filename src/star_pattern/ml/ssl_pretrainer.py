"""Self-supervised pretraining: BYOL-style for astronomical images."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from star_pattern.ml.dataset import FITSDataset, AstroAugmentation
from star_pattern.utils.gpu import get_device
from star_pattern.utils.logging import get_logger

logger = get_logger("ml.ssl")


class BYOLProjector(nn.Module):
    """BYOL projection/prediction head."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BYOL(nn.Module):
    """Bootstrap Your Own Latent (simplified) for self-supervised pretraining."""

    def __init__(
        self,
        backbone: nn.Module,
        embedding_dim: int = 1280,
        hidden_dim: int = 256,
        projection_dim: int = 128,
        momentum: float = 0.996,
    ):
        super().__init__()
        self.online_encoder = backbone
        self.online_projector = BYOLProjector(embedding_dim, hidden_dim, projection_dim)
        self.online_predictor = BYOLProjector(projection_dim, hidden_dim, projection_dim)

        # Target network (EMA of online)
        self.target_encoder = copy.deepcopy(backbone)
        self.target_projector = BYOLProjector(embedding_dim, hidden_dim, projection_dim)

        # Disable gradients for target
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

        self.momentum = momentum

    @torch.no_grad()
    def _update_target(self) -> None:
        """EMA update of target network."""
        for online_p, target_p in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target_p.data = self.momentum * target_p.data + (1 - self.momentum) * online_p.data

        for online_p, target_p in zip(
            self.online_projector.parameters(), self.target_projector.parameters()
        ):
            target_p.data = self.momentum * target_p.data + (1 - self.momentum) * online_p.data

    def forward(
        self, view1: torch.Tensor, view2: torch.Tensor
    ) -> torch.Tensor:
        """Compute BYOL loss from two augmented views."""
        # Online path
        online_feat1 = self.online_encoder(view1)
        if online_feat1.dim() > 2:
            online_feat1 = F.adaptive_avg_pool2d(online_feat1, 1).flatten(1)
        online_proj1 = self.online_projector(online_feat1)
        online_pred1 = self.online_predictor(online_proj1)

        online_feat2 = self.online_encoder(view2)
        if online_feat2.dim() > 2:
            online_feat2 = F.adaptive_avg_pool2d(online_feat2, 1).flatten(1)
        online_proj2 = self.online_projector(online_feat2)
        online_pred2 = self.online_predictor(online_proj2)

        # Target path
        with torch.no_grad():
            target_feat1 = self.target_encoder(view1)
            if target_feat1.dim() > 2:
                target_feat1 = F.adaptive_avg_pool2d(target_feat1, 1).flatten(1)
            target_proj1 = self.target_projector(target_feat1)

            target_feat2 = self.target_encoder(view2)
            if target_feat2.dim() > 2:
                target_feat2 = F.adaptive_avg_pool2d(target_feat2, 1).flatten(1)
            target_proj2 = self.target_projector(target_feat2)

        # Loss: negative cosine similarity
        loss = (
            self._cosine_loss(online_pred1, target_proj2.detach())
            + self._cosine_loss(online_pred2, target_proj1.detach())
        ) / 2

        self._update_target()
        return loss

    @staticmethod
    def _cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        return 2 - 2 * (pred * target).sum(dim=-1).mean()


class SSLPretrainer:
    """Self-supervised pretraining manager."""

    def __init__(
        self,
        data_dir: str,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 3e-4,
        device: str | None = None,
    ):
        self.data_dir = data_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or str(get_device())

    def pretrain(self, save_path: str | None = None) -> dict[str, Any]:
        """Run BYOL pretraining.

        Returns:
            Training history.
        """
        import torchvision.models as models

        # Create two augmentation pipelines
        aug1 = AstroAugmentation(noise_level=0.01)
        aug2 = AstroAugmentation(noise_level=0.03)

        dataset = FITSDataset.from_directory(self.data_dir, transform=aug1)
        if len(dataset) == 0:
            return {"error": "No data"}

        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=True
        )

        # Backbone
        backbone = models.efficientnet_b0(weights=None)
        # Modify for single channel
        backbone.features[0][0] = nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False)
        backbone.classifier = nn.Identity()
        backbone = backbone.to(self.device)

        model = BYOL(backbone, embedding_dim=1280).to(self.device)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        history: dict[str, list[float]] = {"loss": []}
        logger.info(f"BYOL pretraining: {len(dataset)} images, {self.epochs} epochs")

        for epoch in range(self.epochs):
            losses = []
            for batch in loader:
                images = batch["image"].to(self.device)
                # Create two views via augmentation
                view1 = images
                # Simple second view: flip + noise
                view2 = torch.flip(images, dims=[-1]) + 0.02 * torch.randn_like(images)

                loss = model(view1, view2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            scheduler.step()
            epoch_loss = float(np.mean(losses))
            history["loss"].append(epoch_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(f"BYOL Epoch {epoch + 1}/{self.epochs}: loss={epoch_loss:.4f}")

        if save_path:
            torch.save(backbone.state_dict(), save_path)
            logger.info(f"Saved pretrained backbone to {save_path}")

        return history
