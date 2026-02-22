"""Training loop with AdamW + CosineAnnealingLR."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from star_pattern.ml.dataset import FITSDataset, AstroAugmentation
from star_pattern.ml.models import LensNet, AstroClassifier, SimpleUNet
from star_pattern.ml.losses import CombinedLoss, FocalLoss
from star_pattern.utils.gpu import get_device
from star_pattern.utils.logging import get_logger

logger = get_logger("ml.train")


class Trainer:
    """Training loop for astronomical detection models."""

    def __init__(
        self,
        task: str = "lens",
        data_dir: str = "data",
        epochs: int = 100,
        batch_size: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str | None = None,
    ):
        self.task = task
        self.data_dir = data_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device or str(get_device())

        self.model: nn.Module | None = None
        self.criterion: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any = None

    def _setup_model(self, dataset: FITSDataset) -> None:
        """Initialize model, loss, and optimizer based on task."""
        if self.task == "lens":
            self.model = LensNet(input_size=64).to(self.device)
            self.criterion = CombinedLoss()
        elif self.task == "morphology":
            num_classes = len(set(dataset.labels)) if dataset.labels else 5
            self.model = AstroClassifier(embedding_dim=1280, num_classes=num_classes).to(
                self.device
            )
            self.criterion = FocalLoss()
        elif self.task == "anomaly":
            self.model = SimpleUNet(in_channels=1, out_channels=1).to(self.device)
            self.criterion = CombinedLoss()
        else:
            raise ValueError(f"Unknown task: {self.task}")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs
        )

    def run(self, save_dir: str | Path | None = None) -> dict[str, Any]:
        """Run the full training loop.

        Returns:
            Dict with training history (losses, metrics).
        """
        # Create dataset
        augmentation = AstroAugmentation()
        dataset = FITSDataset.from_directory(self.data_dir, transform=augmentation)

        if len(dataset) == 0:
            logger.error(f"No data found in {self.data_dir}")
            return {"error": "No data"}

        # Split train/val (80/20)
        n_val = max(1, int(len(dataset) * 0.2))
        n_train = len(dataset) - n_val
        train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)

        self._setup_model(dataset)

        logger.info(
            f"Training {self.task}: {n_train} train, {n_val} val, {self.epochs} epochs"
        )

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")

        for epoch in range(self.epochs):
            # Train
            self.model.train()
            train_losses = []
            for batch in train_loader:
                images = batch["image"].to(self.device)
                if "label" in batch:
                    targets = torch.tensor(batch["label"], dtype=torch.float32).to(self.device)
                else:
                    targets = images  # Autoencoder-style

                self.optimizer.zero_grad()
                outputs = self.model(images)

                if outputs.shape != targets.shape:
                    targets = targets.view_as(outputs)

                loss = self.criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_losses.append(loss.item())

            self.scheduler.step()

            # Validate
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(self.device)
                    if "label" in batch:
                        targets = torch.tensor(batch["label"], dtype=torch.float32).to(
                            self.device
                        )
                    else:
                        targets = images

                    outputs = self.model(images)
                    if outputs.shape != targets.shape:
                        targets = targets.view_as(outputs)
                    loss = self.criterion(outputs, targets)
                    val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses) if val_losses else 0
            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss))

            if (epoch + 1) % 10 == 0 or epoch == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch + 1}/{self.epochs}: "
                    f"train={train_loss:.4f}, val={val_loss:.4f}, lr={lr:.6f}"
                )

            # Save best
            if val_loss < best_val_loss and save_dir:
                best_val_loss = val_loss
                save_path = Path(save_dir) / f"best_{self.task}_model.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), str(save_path))
                logger.info(f"Saved best model (val_loss={val_loss:.4f})")

        return history
