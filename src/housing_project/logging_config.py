"""Logging helpers shared by workflow scripts."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(
    level: str = "INFO",
    log_path: str | Path | None = None,
    console_log: bool = True,
) -> None:
    """Configure console and optional file logging.

    Parameters
    ----------
    level : str
        Logging level name.
    log_path : str or pathlib.Path, optional
        Destination for a file handler.
    console_log : bool
        Whether to emit log records to stderr.
    """
    handlers: list[logging.Handler] = []
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    if console_log:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=handlers,
        force=True,
    )
