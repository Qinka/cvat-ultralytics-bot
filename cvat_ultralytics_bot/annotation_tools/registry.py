"""Dynamic registry for annotation tool modules.

This module provides a pluggable registry system for annotation tools.
Tools are automatically discovered from the :mod:`cvat_ultralytics_bot.annotation_tools`
package and registered via the :func:`register_tool` function.

The registry follows a plugin-style architecture:
1. Each tool module defines a factory function and calls :func:`register_tool`.
2. On first use, :func:`discover_tools` imports all tool modules.
3. Tools can be retrieved by name using :func:`get_tool_registration`.

Example:
    >>> from cvat_ultralytics_bot.annotation_tools import discover_tools
    >>> discover_tools()
    {'yolo_detect': AnnotationToolRegistration(...), ...}
    >>> from cvat_ultralytics_bot.annotation_tools import get_tool_registration
    >>> tool = get_tool_registration('yolo_detect')
    >>> model = tool.factory({'weights': 'yolov8n.pt'})
"""

from __future__ import annotations

import sys
from importlib import import_module, reload
from pathlib import Path
from pkgutil import iter_modules

from cvat_ultralytics_bot.annotation_tools.base import AnnotationToolRegistration

_REGISTRY: dict[str, AnnotationToolRegistration] = {}
_DISCOVERED = False


def register_tool(registration: AnnotationToolRegistration) -> None:
    """Register an annotation tool.

    Args:
        registration: The tool registration containing name, factory, and metadata.

    Example:
        >>> register_tool(AnnotationToolRegistration(
        ...     name="my_tool",
        ...     factory=my_factory,
        ...     description="My custom tool"
        ... ))
    """
    _REGISTRY[registration.name] = registration


def _iter_tool_modules() -> list[str]:
    """Iterate over discoverable tool module names.

    Returns:
        List of module names (excluding private modules and base/registry).
    """
    package_dir = Path(__file__).resolve().parent
    module_names: list[str] = []
    for module_info in iter_modules([str(package_dir)]):
        if module_info.name.startswith("_") or module_info.name in {"base", "registry", "__init__"}:
            continue
        module_names.append(module_info.name)
    return module_names


def discover_tools(force: bool = False) -> dict[str, AnnotationToolRegistration]:
    """Import tool modules once and return the registry cache.

    This function automatically discovers and imports all tool modules
    in the :mod:`cvat_ultralytics_bot.annotation_tools` package.
    It is called internally by :func:`get_tool_registration` and
    :func:`list_tool_registrations`.

    Args:
        force: If True, reload all modules even if already discovered.

    Returns:
        Dictionary mapping tool names to their registrations.

    Note:
        This function only imports modules; it does not instantiate
        any tools. Tool instantiation is done via the factory in
        each registration.
    """
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
    """Return a registered tool by name.

    Args:
        name: Name of the tool to retrieve.

    Returns:
        The tool's registration object.

    Raises:
        ValueError: If the tool name is not registered.
    """
    registry = discover_tools()
    try:
        return registry[name]
    except KeyError as exc:
        available = ", ".join(sorted(registry)) or "<none>"
        raise ValueError(f"Unknown annotation tool '{name}'. Available tools: {available}") from exc


def list_tool_registrations() -> dict[str, AnnotationToolRegistration]:
    """Return all discovered tool registrations.

    Returns:
        Dictionary mapping tool names to their registrations.
    """
    return discover_tools()
