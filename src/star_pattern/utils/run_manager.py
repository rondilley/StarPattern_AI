"""Run management: directories, checkpoints, and state persistence."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from star_pattern.utils.logging import get_logger

logger = get_logger("run_manager")


class RunManager:
    """Manages run directories, checkpointing, and state persistence."""

    def __init__(self, base_dir: str | Path = "output/runs", run_name: str | None = None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name
        self.run_dir = self.base_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.images_dir = self.run_dir / "images"
        self.results_dir = self.run_dir / "results"
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.reports_dir = self.run_dir / "reports"

        for d in [self.images_dir, self.results_dir, self.checkpoints_dir, self.reports_dir]:
            d.mkdir(exist_ok=True)

        self.state: dict[str, Any] = {"run_name": run_name, "created": run_name}
        self._state_file = self.run_dir / "state.json"
        if self._state_file.exists():
            self.state = json.loads(self._state_file.read_text())
        else:
            self._save_state()

        logger.info(f"Run directory: {self.run_dir}")

    def _save_state(self) -> None:
        self._state_file.write_text(json.dumps(self.state, indent=2, default=str))

    def save_checkpoint(self, name: str, data: dict[str, Any]) -> Path:
        """Save a checkpoint."""
        path = self.checkpoints_dir / f"{name}.json"
        path.write_text(json.dumps(data, indent=2, default=str))
        self.state["last_checkpoint"] = name
        self._save_state()
        logger.info(f"Saved checkpoint: {name}")
        return path

    def load_checkpoint(self, name: str) -> dict[str, Any] | None:
        """Load a checkpoint by name."""
        path = self.checkpoints_dir / f"{name}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def save_result(self, name: str, data: dict[str, Any]) -> Path:
        """Save a result file."""
        path = self.results_dir / f"{name}.json"
        path.write_text(json.dumps(data, indent=2, default=str))
        return path

    def load_result(self, name: str) -> dict[str, Any] | None:
        """Load a result by name."""
        path = self.results_dir / f"{name}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def save_image(self, name: str, data: Any) -> Path:
        """Save an image (matplotlib figure or numpy array)."""
        path = self.images_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)

        import matplotlib

        if isinstance(data, matplotlib.figure.Figure):
            data.savefig(str(path), dpi=150, bbox_inches="tight")
        else:
            import numpy as np
            from PIL import Image

            if isinstance(data, np.ndarray):
                Image.fromarray(data).save(str(path))
            else:
                raise TypeError(f"Unsupported image type: {type(data)}")

        return path

    def update_state(self, **kwargs: Any) -> None:
        """Update run state with arbitrary key-value pairs."""
        self.state.update(kwargs)
        self._save_state()

    @classmethod
    def latest(cls, base_dir: str | Path = "output/runs") -> RunManager | None:
        """Load the most recent run."""
        base = Path(base_dir)
        if not base.exists():
            return None
        runs = sorted(base.iterdir())
        if not runs:
            return None
        return cls(base_dir=base_dir, run_name=runs[-1].name)

    def __repr__(self) -> str:
        return f"RunManager({self.run_dir})"
