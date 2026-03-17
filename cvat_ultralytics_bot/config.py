"""Load annotation and connection config files.

This module provides configuration loading for the CVAT bot:
- :class:`ConnectionConfig`: Dataclass for CVAT connection credentials.
- :class:`AnnotationConfig`: Dataclass for annotation task settings.
- Functions to encrypt/decrypt connection configs.
- Functions to load TOML annotation configs.

Example:
    >>> from cvat_ultralytics_bot.config import load_connection_config
    >>> conn = load_connection_config("connection.json")
    >>> print(conn.host)
    http://localhost:8080
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found]


from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ConnectionConfig:
    """Decrypted CVAT connection settings.

    Attributes:
        host: CVAT server URL (e.g., ``http://localhost:8080``).
        username: CVAT account username.
        password: CVAT account password.
    """

    host: str
    username: str
    password: str


@dataclass(frozen=True)
class AnnotationConfig:
    """Annotation execution settings loaded from TOML.

    Attributes:
        tool: Name of the annotation tool to use.
        conf: Confidence threshold for predictions.
        device: Device for model inference (e.g., ``"cpu"``, ``"cuda:0"``).
        replace: Whether to replace existing annotations.
        frame_ids: Specific frame IDs to annotate, or None for all.
        label_map: Mapping from model class names to CVAT label names.
        tool_config: Tool-specific configuration dictionary.
    """

    tool: str
    conf: float
    device: str
    replace: bool
    frame_ids: list[int] | None
    label_map: dict[str, str] | None
    tool_config: dict[str, Any]


def _checksum_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _xor_bytes(data: bytes, key: bytes) -> bytes:
    return bytes(byte ^ key[index % len(key)] for index, byte in enumerate(data))


def encrypt_connection_payload(payload: dict[str, str]) -> tuple[str, str]:
    """Obfuscate a JSON payload without an external secret.

    This is lightweight reversible obfuscation based on SHA-256 derived bytes
    and base64. It avoids storing credentials in clear text, but is not a
    substitute for real encryption or a secret manager.

    Args:
        payload: Dictionary containing connection details (host, username, password).

    Returns:
        Tuple of (encrypted_payload, checksum_hex).
    """
    raw = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    checksum = _checksum_bytes(raw)
    encrypted = _xor_bytes(raw, checksum)
    return base64.urlsafe_b64encode(encrypted).decode("ascii"), checksum.hex()


def decrypt_connection_payload(cipher_text: str, checksum_hex: str) -> dict[str, str]:
    """De-obfuscate a connection payload and verify its checksum.

    Args:
        cipher_text: Base64-encoded encrypted payload.
        checksum_hex: Hex-encoded SHA256 checksum for verification.

    Returns:
        Dictionary with decrypted connection details.

    Raises:
        ValueError: If checksum verification fails.
    """
    encrypted = base64.urlsafe_b64decode(cipher_text.encode("ascii"))
    checksum = bytes.fromhex(checksum_hex)
    raw = _xor_bytes(encrypted, checksum)
    if _checksum_bytes(raw).hex() != checksum_hex:
        logger.error("Connection config checksum verification failed")
        raise ValueError("Connection config checksum verification failed")
    return json.loads(raw.decode("utf-8"))


def dump_connection_config(path: str | Path, host: str, username: str, password: str) -> Path:
    """Write an obfuscated connection config file.

    Args:
        path: Output file path.
        host: CVAT server URL.
        username: CVAT account username.
        password: CVAT account password.

    Returns:
        Path to the created config file.
    """
    logger.debug("Creating connection config at: %s", path)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encrypted, checksum = encrypt_connection_payload(
        {"host": host, "username": username, "password": password}
    )
    document = {
        "version": 2,
        "encryption": "sha256-xor-base64",
        "checksum": checksum,
        "payload": encrypted,
    }
    target.write_text(json.dumps(document, indent=2), encoding="utf-8")
    logger.info("Connection config written to: %s", target)
    return target


def load_connection_config(path: str | Path) -> ConnectionConfig:
    """Read and de-obfuscate a connection config file.

    Args:
        path: Path to the config file.

    Returns:
        ConnectionConfig with decrypted credentials.

    Raises:
        ValueError: If config file is invalid or checksum fails.
    """
    logger.debug("Loading connection config from: %s", path)
    source = Path(path)
    document = json.loads(source.read_text(encoding="utf-8"))
    payload = decrypt_connection_payload(document["payload"], document["checksum"])
    logger.debug("Connection config loaded for host: %s", payload.get("host"))
    return ConnectionConfig(**payload)


def _parse_label_map(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("label_map must be a TOML table")
    return {str(key): str(mapped) for key, mapped in value.items()}


def _get_config_value(config: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in config:
            return config[key]
    return None


def _parse_frame_ids(value: Any) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("frames must be a TOML array of integers")
    return [int(frame_id) for frame_id in value]


def load_annotation_config(path: str | Path) -> AnnotationConfig:
    """Load an annotation config TOML file."""
    source = Path(path)
    document = tomllib.loads(source.read_text(encoding="utf-8"))
    tool = document.get("tool")
    if not isinstance(tool, str) or not tool:
        raise ValueError("annotation config must define a non-empty 'tool'")

    tool_section = document.get(tool, {})
    if not isinstance(tool_section, dict):
        raise ValueError(f"annotation config section '[{tool}]' must be a table")

    shared_keys = {"conf", "device", "replace", "frames", "label_map", "label-map", "use_polygon"}
    tool_config = {
        key: value for key, value in tool_section.items() if key not in shared_keys
    }
    # Also add label_map to tool_config if present (for tools that handle it directly)
    section_label_map = _get_config_value(tool_section, "label_map", "label-map")
    root_label_map = _get_config_value(document, "label_map", "label-map")
    tool_config["label_map"] = section_label_map if section_label_map is not None else root_label_map

    # Add use_polygon to tool_config if present
    section_use_polygon = _get_config_value(tool_section, "use_polygon")
    root_use_polygon = _get_config_value(document, "use_polygon")
    if section_use_polygon is not None:
        tool_config["use_polygon"] = section_use_polygon
    elif root_use_polygon is not None:
        tool_config["use_polygon"] = root_use_polygon
    section_frames = _get_config_value(tool_section, "frames")
    root_frames = _get_config_value(document, "frames")
    section_conf = _get_config_value(tool_section, "conf")
    root_conf = _get_config_value(document, "conf")
    section_device = _get_config_value(tool_section, "device")
    root_device = _get_config_value(document, "device")
    section_replace = _get_config_value(tool_section, "replace")
    root_replace = _get_config_value(document, "replace")

    return AnnotationConfig(
        tool=tool,
        conf=float(section_conf if section_conf is not None else root_conf if root_conf is not None else 0.25),
        device=str(section_device if section_device is not None else root_device if root_device is not None else "cpu"),
        replace=bool(section_replace if section_replace is not None else root_replace if root_replace is not None else False),
        frame_ids=_parse_frame_ids(section_frames if section_frames is not None else root_frames),
        label_map=_parse_label_map(section_label_map if section_label_map is not None else root_label_map),
        tool_config=tool_config,
    )
