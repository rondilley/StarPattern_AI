"""Structured logging setup."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


_configured = False


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Configure structured logging for the application."""
    global _configured
    if _configured:
        return

    root = logging.getLogger("star_pattern")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file))
        fh.setFormatter(fmt)
        root.addHandler(fh)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger within the star_pattern namespace."""
    if not _configured:
        setup_logging()
    return logging.getLogger(f"star_pattern.{name}")
