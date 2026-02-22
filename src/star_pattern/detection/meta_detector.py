"""Learned meta-detector: non-linear cross-detector scoring via active learning.

Progression based on label count:
1. Linear baseline (0-49 labels): identical to weighted ensemble
2. Gradient boosting (50+ labels): sklearn GBM with predict_proba
3. Small neural net (200+ labels): PyTorch MLP [n_features, 64, 32, 1]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from star_pattern.utils.logging import get_logger

logger = get_logger("detection.meta_detector")


@dataclass
class MetaDetectorConfig:
    """Configuration for the learned meta-detector."""

    enabled: bool = True
    blend_weight: float = 0.0  # 0=all linear, 1=all learned (genome-evolvable)
    min_samples_gbm: int = 50
    min_samples_nn: int = 200
    gbm_n_estimators: int = 100
    gbm_max_depth: int = 3
    nn_hidden: list[int] = field(default_factory=lambda: [64, 32])


class MetaDetector:
    """Learned meta-detector that combines cross-detector features non-linearly.

    Starts as a linear blend (identical to current ensemble), then
    progressively upgrades to GBM and neural net as labeled data accumulates.
    """

    def __init__(self, config: MetaDetectorConfig | None = None):
        self.config = config or MetaDetectorConfig()
        self._features: list[np.ndarray] = []
        self._labels: list[bool] = []
        self._gbm_model = None
        self._nn_model = None
        self._scaler = None
        self._model_type = "linear"
        self._feature_importance: dict[str, float] = {}

    @property
    def n_samples(self) -> int:
        return len(self._labels)

    @property
    def model_type(self) -> str:
        return self._model_type

    def score(
        self, features: np.ndarray, linear_score: float
    ) -> dict[str, Any]:
        """Score a detection using the best available model.

        Args:
            features: Rich feature vector from FeatureFusionExtractor.
            linear_score: Original linear ensemble anomaly_score.

        Returns:
            Dict with meta_score, learned_score, linear_score,
            feature_importance, and model_type.
        """
        blend = self.config.blend_weight
        learned_score = self._learned_score(features)

        meta_score = (1.0 - blend) * linear_score + blend * learned_score

        return {
            "meta_score": float(np.clip(meta_score, 0, 1)),
            "learned_score": float(np.clip(learned_score, 0, 1)),
            "linear_score": float(linear_score),
            "blend_weight": blend,
            "model_type": self._model_type,
            "feature_importance": dict(self._feature_importance),
        }

    def _learned_score(self, features: np.ndarray) -> float:
        """Compute the learned model's score."""
        features_2d = features.reshape(1, -1)

        # Neural net (highest priority)
        if self._nn_model is not None and self._scaler is not None:
            try:
                return self._score_nn(features_2d)
            except Exception as e:
                logger.debug(f"NN scoring failed: {e}")

        # Gradient boosting
        if self._gbm_model is not None and self._scaler is not None:
            try:
                return self._score_gbm(features_2d)
            except Exception as e:
                logger.debug(f"GBM scoring failed: {e}")

        # Fallback: use the linear score directly
        return float(features[features.size - 3]) if features.size >= 3 else 0.5

    def _score_gbm(self, features_2d: np.ndarray) -> float:
        """Score with gradient boosting model."""
        scaled = self._scaler.transform(features_2d)
        proba = self._gbm_model.predict_proba(scaled)[0]
        # proba[1] = P(interesting)
        return float(proba[1]) if len(proba) > 1 else float(proba[0])

    def _score_nn(self, features_2d: np.ndarray) -> float:
        """Score with neural net model."""
        import torch

        scaled = self._scaler.transform(features_2d)
        tensor = torch.tensor(scaled, dtype=torch.float32)
        self._nn_model.eval()
        with torch.no_grad():
            logit = self._nn_model(tensor)
            prob = torch.sigmoid(logit).item()
        return prob

    def add_sample(
        self, features: np.ndarray, is_interesting: bool
    ) -> None:
        """Record a labeled sample for future retraining.

        Args:
            features: Rich feature vector.
            is_interesting: Whether this detection was labeled interesting.
        """
        self._features.append(features.copy())
        self._labels.append(is_interesting)

    def retrain(self) -> dict[str, Any]:
        """Retrain the best model given current label count.

        Returns:
            Dict with model_type, n_samples, and training metrics.
        """
        n = self.n_samples
        if n < 10:
            return {"model_type": "linear", "n_samples": n, "status": "insufficient_data"}

        X = np.stack(self._features)
        y = np.array(self._labels, dtype=np.float64)

        # Fit scaler
        from sklearn.preprocessing import StandardScaler
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        result: dict[str, Any] = {"n_samples": n}

        # Try neural net first (200+ samples)
        if n >= self.config.min_samples_nn:
            try:
                metrics = self._train_nn(X_scaled, y)
                self._model_type = "neural_net"
                result.update({"model_type": "neural_net", **metrics})
                logger.info(f"Meta-detector retrained: neural_net ({n} samples)")
                self._compute_feature_importance(X_scaled, y)
                return result
            except Exception as e:
                logger.warning(f"NN training failed, falling back to GBM: {e}")

        # Try GBM (50+ samples)
        if n >= self.config.min_samples_gbm:
            try:
                metrics = self._train_gbm(X_scaled, y)
                self._model_type = "gbm"
                result.update({"model_type": "gbm", **metrics})
                logger.info(f"Meta-detector retrained: gbm ({n} samples)")
                self._compute_feature_importance(X_scaled, y)
                return result
            except Exception as e:
                logger.warning(f"GBM training failed: {e}")

        self._model_type = "linear"
        result["model_type"] = "linear"
        return result

    def _train_gbm(
        self, X_scaled: np.ndarray, y: np.ndarray
    ) -> dict[str, float]:
        """Train gradient boosting classifier."""
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score

        model = GradientBoostingClassifier(
            n_estimators=self.config.gbm_n_estimators,
            max_depth=self.config.gbm_max_depth,
            learning_rate=0.1,
            random_state=42,
        )
        model.fit(X_scaled, y.astype(int))
        self._gbm_model = model

        # Cross-validation score if enough data
        cv_score = 0.0
        if len(y) >= 20:
            try:
                cv_scores = cross_val_score(
                    model, X_scaled, y.astype(int), cv=min(5, len(y) // 5),
                    scoring="roc_auc",
                )
                cv_score = float(np.mean(cv_scores))
            except Exception:
                pass

        return {"cv_auc": cv_score}

    def _train_nn(
        self, X_scaled: np.ndarray, y: np.ndarray
    ) -> dict[str, float]:
        """Train small MLP classifier with train/val split and early stopping."""
        import torch
        import torch.nn as nn
        from sklearn.model_selection import train_test_split

        n_features = X_scaled.shape[1]
        hidden = self.config.nn_hidden

        # 80/20 train/validation split (stratified to preserve class balance)
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42,
                stratify=y.astype(int),
            )
        except ValueError:
            # Stratification fails if a class has < 2 samples
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42,
            )

        # Build MLP
        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        model = nn.Sequential(*layers)

        # Tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        # Early stopping state
        best_val_loss = float("inf")
        best_state = None
        patience = 10
        patience_counter = 0
        max_epochs = 100

        train_losses = []
        val_losses = []

        for epoch in range(max_epochs):
            # Train
            model.train()
            optimizer.zero_grad()
            logits = model(X_train_t)
            loss = criterion(logits, y_train_t)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            # Validate
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t)
                val_loss = criterion(val_logits, y_val_t).item()
            val_losses.append(val_loss)

            # Early stopping check
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.debug(
                        f"Early stopping at epoch {epoch + 1}, "
                        f"best val_loss={best_val_loss:.4f}"
                    )
                    break

        # Restore best model weights
        if best_state is not None:
            model.load_state_dict(best_state)

        self._nn_model = model
        return {
            "final_train_loss": float(train_losses[-1]),
            "best_val_loss": float(best_val_loss),
            "epochs_trained": len(train_losses),
        }

    def _compute_feature_importance(
        self, X_scaled: np.ndarray, y: np.ndarray
    ) -> None:
        """Compute feature importance using the GBM model or permutation."""
        if self._gbm_model is not None:
            importances = self._gbm_model.feature_importances_
            self._feature_importance = {
                str(i): float(v) for i, v in enumerate(importances)
            }

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance dict (index -> importance)."""
        return dict(self._feature_importance)

    def save_state(self, path: Path | str) -> None:
        """Save meta-detector state to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save samples
        if self._features:
            np.save(str(path / "features.npy"), np.stack(self._features))
            np.save(str(path / "labels.npy"), np.array(self._labels))

        # Save metadata
        meta = {
            "model_type": self._model_type,
            "n_samples": self.n_samples,
            "feature_importance": self._feature_importance,
            "config": {
                "blend_weight": self.config.blend_weight,
                "min_samples_gbm": self.config.min_samples_gbm,
                "min_samples_nn": self.config.min_samples_nn,
                "gbm_n_estimators": self.config.gbm_n_estimators,
                "gbm_max_depth": self.config.gbm_max_depth,
                "nn_hidden": self.config.nn_hidden,
            },
        }
        (path / "meta_detector.json").write_text(json.dumps(meta, indent=2))

        # Save sklearn model
        if self._gbm_model is not None:
            try:
                import joblib
                joblib.dump(self._gbm_model, str(path / "gbm_model.joblib"))
                if self._scaler is not None:
                    joblib.dump(self._scaler, str(path / "scaler.joblib"))
            except ImportError:
                pass

        # Save PyTorch model
        if self._nn_model is not None:
            try:
                import torch
                torch.save(self._nn_model.state_dict(), str(path / "nn_model.pt"))
            except Exception as e:
                logger.debug(f"Failed to save NN model: {e}")

        logger.info(f"Meta-detector state saved to {path}")

    def load_state(self, path: Path | str) -> None:
        """Load meta-detector state from disk."""
        path = Path(path)
        if not path.exists():
            return

        # Load samples
        feat_path = path / "features.npy"
        label_path = path / "labels.npy"
        if feat_path.exists() and label_path.exists():
            features = np.load(str(feat_path))
            labels = np.load(str(label_path))
            self._features = [features[i] for i in range(len(features))]
            self._labels = [bool(labels[i]) for i in range(len(labels))]

        # Load metadata
        meta_path = path / "meta_detector.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self._model_type = meta.get("model_type", "linear")
            self._feature_importance = meta.get("feature_importance", {})

        # Load sklearn model
        gbm_path = path / "gbm_model.joblib"
        scaler_path = path / "scaler.joblib"
        if gbm_path.exists():
            try:
                import joblib
                self._gbm_model = joblib.load(str(gbm_path))
                if scaler_path.exists():
                    self._scaler = joblib.load(str(scaler_path))
            except ImportError:
                logger.debug("joblib not available, cannot load GBM model")

        logger.info(
            f"Meta-detector state loaded: {self._model_type} "
            f"({self.n_samples} samples)"
        )
