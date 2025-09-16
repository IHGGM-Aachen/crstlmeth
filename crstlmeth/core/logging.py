"""
crstlmeth/core/logging.py

simple tsv-based logging for crstlmeth command-line tools
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click

__all__ = ["get_logger", "get_logger_from_cli", "log_event"]


_LOGGERS: dict[str, logging.Logger] = {}


def _ts() -> str:
    """
    return a utc timestamp string in iso-8601 format (to seconds precision)
    """
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class _TSVFormatter(logging.Formatter):
    """
    log formatter that emits a fixed-column tsv line for each record

    columns:
        ts, level, session, event, command, parameters, message, runtime
    """

    def format(self, record: logging.LogRecord) -> str:
        ts = _ts()
        lvl = record.levelname
        session = getattr(record, "session", "local")
        event = getattr(record, "event", "-")
        cmd = getattr(record, "cmd", "-")
        params = getattr(record, "params", {})
        msg = record.getMessage()
        runtime = getattr(record, "runtime", "")
        return "\t".join(
            [
                ts,
                lvl,
                session,
                event,
                cmd,
                json.dumps(params, separators=(",", ":")),
                msg.replace("\t", " "),
                str(runtime),
            ]
        )


def get_logger(
    log_path: str | Path, session_id: str = "local"
) -> logging.Logger:
    """
    create or return a per-file singleton logger instance

    the logger writes to the specified tsv file and reuses the same
    object for subsequent calls with the same path.
    """
    # normalise path; if it's a directory, drop a default filename inside it
    p = Path(log_path).expanduser().resolve()
    if p.is_dir():
        p = p / "crstlmeth.log.tsv"

    # ensure parent exists
    p.parent.mkdir(parents=True, exist_ok=True)

    key = str(p)
    if key in _LOGGERS:
        return _LOGGERS[key]

    logger = logging.getLogger(f"crstlmeth.{key}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = logging.FileHandler(p, encoding="utf-8")
    handler.setFormatter(_TSVFormatter())
    logger.addHandler(handler)

    logger.session_id = session_id  # type: ignore[attr-defined]

    _LOGGERS[key] = logger
    return logger


def get_logger_from_cli(ctx: click.Context) -> logging.Logger:
    """
    convenience helper: return a logger using the context's --log-file,
    or fall back to a file in the current working directory
    """
    if (
        ctx is not None
        and ctx.obj
        and "log_file" in ctx.obj
        and ctx.obj["log_file"]
    ):
        log_path = Path(ctx.obj["log_file"])
    else:
        # IMPORTANT: fallback must be a *file*, not a directory
        log_path = Path.cwd() / "crstlmeth.log.tsv"
    return get_logger(log_path)


def log_event(
    logger: logging.Logger,
    *,
    level: int = logging.INFO,
    event: str,
    cmd: str,
    params: dict[str, Any],
    message: str = "ok",
    runtime_s: float | None = None,
) -> None:
    """
    emit a canonical tsv log line using the provided logger

    parameters:
        logger     : result of get_logger()
        level      : logging.INFO, WARNING or ERROR
        event      : short tag (e.g. "analyze", "calculate-meth")
        cmd        : raw cli command or internal function
        params     : dictionary of user arguments (json-encoded)
        message    : status message or error description
        runtime_s  : optional elapsed time in seconds
    """
    extra = {
        "session": getattr(logger, "session_id", "local"),
        "event": event,
        "cmd": cmd,
        "params": params,
        "runtime": f"{runtime_s:.2f}" if runtime_s is not None else "",
    }
    logger.log(level, message, extra=extra)
