"""Compatibility imports; graph validation belongs to core and runtime."""
from ..core.topology import validate_topology  # noqa: F401
from ..runtime.validation import validate_capabilities  # noqa: F401
