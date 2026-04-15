from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path

_CONFIGURED = False


def _configure_root_logger() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    project_root = Path(__file__).resolve().parent.parent
    log_dir = project_root / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "trading_selector.log"

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger("trading_selector")
    env_level = os.getenv("TRADING_SELECTOR_LOG_LEVEL", "INFO").upper().strip()
    root.setLevel(getattr(logging, env_level, logging.INFO))
    root.propagate = False

    if not root.handlers:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=4 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        root.addHandler(stream_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure_root_logger()
    return logging.getLogger(f"trading_selector.{name}")


def set_log_level(level_name: str) -> None:
    """运行时修改日志级别（INFO/DEBUG/WARNING/ERROR）。"""
    _configure_root_logger()
    root = logging.getLogger("trading_selector")
    level = getattr(logging, str(level_name).upper().strip(), logging.INFO)
    root.setLevel(level)
    for h in root.handlers:
        h.setLevel(level)
