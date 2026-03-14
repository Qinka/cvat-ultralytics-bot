"""Dynamic registry for annotation tool modules."""

from __future__ import annotations

import sys
from importlib import import_module, reload
from pathlib import Path
from pkgutil import iter_modules

from cvat_ultralytics_bot.annotation_tools.base import AnnotationToolRegistration

_REGISTRY: dict[str, AnnotationToolRegistration] = {}
_DISCOVERED = False


def register_tool(registration: AnnotationToolRegistration) -> None:
    """Register an annotation tool."""
    _REGISTRY[registration.name] = registration


def _iter_tool_modules() -> list[str]:
    package_dir = Path(__file__).resolve().parent
    module_names: list[str] = []
    for module_info in iter_modules([str(package_dir)]):
        if module_info.name.startswith("_") or module_info.name in {"base", "registry", "__init__"}:
            continue
        module_names.append(module_info.name)
    return module_names


def discover_tools(force: bool = False) -> dict[str, AnnotationToolRegistration]:
    """Import tool modules once and return the registry cache."""
    global _DISCOVERED
    if _DISCOVERED and not force:
        return dict(_REGISTRY)

    if force:
        _REGISTRY.clear()

    for module_name in _iter_tool_modules():
        qualified_name = f"cvat_ultralytics_bot.annotation_tools.{module_name}"
        if force and qualified_name in sys.modules:
            reload(sys.modules[qualified_name])
        else:
            import_module(qualified_name)

    _DISCOVERED = True
    return dict(_REGISTRY)


def get_tool_registration(name: str) -> AnnotationToolRegistration:
    """Return a registered tool by name."""
    registry = discover_tools()
    try:
        return registry[name]
    except KeyError as exc:
        available = ", ".join(sorted(registry)) or "<none>"
        raise ValueError(f"Unknown annotation tool '{name}'. Available tools: {available}") from exc


def list_tool_registrations() -> dict[str, AnnotationToolRegistration]:
    """Return all discovered tool registrations."""
    return discover_tools()
