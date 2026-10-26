"""Environment-based configuration for PlantPilot.

Secrets are loaded from environment variables only. The ``.env`` file is
loaded automatically by ``python-dotenv`` if present, but the actual
runtime values come from the process environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env if it exists. This never overrides values already present in
# the process environment, so explicit exports still take precedence.
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "plantpilot.db"


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(
            f"Environment variable {name!r} must be an integer, got {raw!r}"
        ) from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(
            f"Environment variable {name!r} must be a number, got {raw!r}"
        ) from exc


@dataclass
class Config:
    """Runtime configuration, resolved from environment variables."""

    nvidia_api_key: str
    nvidia_base_url: str
    model_identifier: str
    database_path: Path
    max_steps: int
    max_tool_calls: int
    max_input_size: int
    max_output_tokens: int
    api_timeout_seconds: float
    tool_result_max_bytes: int
    tool_result_max_chars: int
    evidence_max_chars: int
    evidence_max_bytes: int
    max_message_chars: int

    @property
    def data_dir(self) -> Path:
        return self.database_path.parent

    def ensure_data_dir(self) -> None:
        """Create the data directory if it does not exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)


def load_config() -> Config:
    """Build a :class:`Config` from the current environment.

    The NVIDIA API key is required. If it is absent, raise
    :class:`ConfigError` rather than silently using an empty string.
    """
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise ConfigError(
            "NVIDIA_API_KEY is not set. Create a .env file or export the "
            "variable before starting PlantPilot."
        )

    db_path = _env_str("PLANTPILOT_DATABASE_PATH", str(DEFAULT_DB_PATH))
    db_path_obj = Path(db_path).expanduser()

    return Config(
        nvidia_api_key=api_key,
        nvidia_base_url=_env_str(
            "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
        ),
        model_identifier=_env_str(
            "PLANTPILOT_MODEL", "nvidia/nemotron-3.5-super-vl-120b-a12b"
        ),
        database_path=db_path_obj,
        max_steps=_env_int("PLANTPILOT_MAX_STEPS", 8),
        max_tool_calls=_env_int("PLANTPILOT_MAX_TOOL_CALLS", 20),
        max_input_size=_env_int("PLANTPILOT_MAX_INPUT_SIZE", 8000),
        max_output_tokens=_env_int("PLANTPILOT_MAX_OUTPUT_TOKENS", 4000),
        api_timeout_seconds=_env_float("PLANTPILOT_API_TIMEOUT_SECONDS", 60.0),
        tool_result_max_bytes=_env_int("PLANTPILOT_TOOL_RESULT_MAX_BYTES", 4000),
        tool_result_max_chars=_env_int("PLANTPILOT_TOOL_RESULT_MAX_CHARS", 1200),
        evidence_max_chars=_env_int("PLANTPILOT_EVIDENCE_MAX_CHARS", 2000),
        evidence_max_bytes=_env_int("PLANTPILOT_EVIDENCE_MAX_BYTES", 8000),
        max_message_chars=_env_int("PLANTPILOT_MAX_MESSAGE_CHARS", 12000),
    )


_config: Optional[Config] = None


def get_config() -> Config:
    """Return the cached configuration, loading it on first access."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    """Clear the cached configuration. Useful for tests."""
    global _config
    _config = None