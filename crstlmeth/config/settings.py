# crstlmeth/config/settings.py
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path | None = None
    out_dir: Path | None = None
    refs_dir: Path | None = None
    regions_dir: Path | None = None
    tabix: str = "tabix"
    default_kit: str = "ME030"
    web_port: int = 8501


def load_settings(config_path: str | Path | None = None) -> Settings:
    """
    Load Settings from TOML.

    Search order:
      1) explicit config_path argument
      2) env var CRSTLMETH_CONFIG
      3) ./crstlmeth.toml
      4) ~/.config/crstlmeth/config.toml

    Missing config -> defaults.
    """
    if config_path is None:
        config_path = os.getenv("CRSTLMETH_CONFIG")

    candidates: list[Path] = []
    if config_path:
        candidates.append(Path(str(config_path)))

    candidates += [
        Path.cwd() / "crstlmeth.toml",
        Path.home() / ".config" / "crstlmeth" / "config.toml",
    ]

    cfg: dict = {}
    for p in candidates:
        try:
            if p.exists():
                with p.open("rb") as f:
                    cfg = tomllib.load(f)
                break
        except Exception:
            # malformed/unreadable config -> ignore and try next
            continue

    paths = cfg.get("paths", {}) or {}
    defaults = cfg.get("defaults", {}) or {}
    web = cfg.get("web", {}) or {}

    def _p(x: object) -> Path | None:
        if not x:
            return None
        return Path(str(x)).expanduser().resolve()

    def _s(x: object, fallback: str) -> str:
        v = str(x).strip() if x is not None else ""
        return v or fallback

    return Settings(
        data_dir=_p(paths.get("data_dir")),
        out_dir=_p(paths.get("out_dir")),
        refs_dir=_p(paths.get("refs_dir")),
        regions_dir=_p(paths.get("regions_dir")),
        tabix=_s(paths.get("tabix"), "tabix"),
        default_kit=_s(defaults.get("kit"), "ME030"),
        web_port=int(web.get("port", 8501)),
    )
