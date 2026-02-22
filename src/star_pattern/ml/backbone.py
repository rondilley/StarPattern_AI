"""Frozen backbone wrapper for ZooBot/AstroCLIP/torchvision models."""

from __future__ import annotations

from typing import Any

import numpy as np

from star_pattern.utils.logging import get_logger
from star_pattern.utils.gpu import get_device

logger = get_logger("ml.backbone")


class BackboneWrapper:
    """Wrapper for frozen feature extraction backbones.

    Supports: torchvision (ResNet, EfficientNet), ZooBot, custom checkpoints.
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        device: str | None = None,
    ):
        self.model_name = model_name
        self._device = device
        self._model = None
        self._transform = None
        self._embedding_dim: int | None = None
        self._init_model(pretrained)

    def _init_model(self, pretrained: bool) -> None:
        """Initialize the backbone model."""
        import torch
        import torchvision.models as models
        from torchvision import transforms

        device = self._device or str(get_device())

        if self.model_name.startswith("zoobot"):
            self._init_zoobot(device)
            return

        # Torchvision models
        weights = "DEFAULT" if pretrained else None

        if self.model_name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=weights)
            self._embedding_dim = 1280
            # Remove classifier
            model.classifier = torch.nn.Identity()
        elif self.model_name == "resnet50":
            model = models.resnet50(weights=weights)
            self._embedding_dim = 2048
            model.fc = torch.nn.Identity()
        elif self.model_name == "resnet18":
            model = models.resnet18(weights=weights)
            self._embedding_dim = 512
            model.fc = torch.nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {self.model_name}")

        model = model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        self._model = model
        self._device = device

        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        logger.info(f"Loaded backbone: {self.model_name} (dim={self._embedding_dim})")

    def _init_zoobot(self, device: str) -> None:
        """Initialize ZooBot backbone (if installed)."""
        try:
            from zoobot.pytorch.training import finetune

            model = finetune.FinetuneableZoobotClassifier(
                name="hf_hub:mwalmsley/zoobot-encoder-convnext_nano",
                num_classes=2,
            )
            self._model = model.encoder
            self._model.to(device)
            self._model.eval()
            self._embedding_dim = 640  # ConvNeXt nano
            self._device = device

            from torchvision import transforms

            self._transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ]
            )
            logger.info("Loaded ZooBot backbone")
        except ImportError:
            logger.warning("ZooBot not installed, falling back to EfficientNet")
            self.model_name = "efficientnet_b0"
            self._init_model(pretrained=True)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim or 1280

    def embed_image(self, image: np.ndarray) -> np.ndarray:
        """Extract embedding from a single image.

        Args:
            image: 2D (grayscale) or 3D (H,W,C) array.

        Returns:
            1D embedding vector.
        """
        import torch

        if image.ndim == 2:
            # Grayscale -> 3-channel
            image = np.stack([image, image, image], axis=-1)

        # Normalize to [0, 255] uint8
        if image.dtype == np.float32 or image.dtype == np.float64:
            vmin, vmax = np.percentile(image, [1, 99])
            image = np.clip((image - vmin) / max(vmax - vmin, 1e-10), 0, 1)
            image = (image * 255).astype(np.uint8)

        tensor = self._transform(image).unsqueeze(0).to(self._device)

        with torch.no_grad():
            embedding = self._model(tensor)

        return embedding.cpu().numpy().flatten()

    def embed_batch(self, images: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """Extract embeddings for a batch of images.

        Returns:
            NxD embedding matrix.
        """
        import torch

        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            tensors = []
            for img in batch:
                if img.ndim == 2:
                    img = np.stack([img, img, img], axis=-1)
                if img.dtype != np.uint8:
                    vmin, vmax = np.percentile(img, [1, 99])
                    img = np.clip((img - vmin) / max(vmax - vmin, 1e-10), 0, 1)
                    img = (img * 255).astype(np.uint8)
                tensors.append(self._transform(img))

            batch_tensor = torch.stack(tensors).to(self._device)
            with torch.no_grad():
                embeddings = self._model(batch_tensor)
            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)
