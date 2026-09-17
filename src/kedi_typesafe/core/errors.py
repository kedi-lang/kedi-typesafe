from __future__ import annotations


class TypeSafeIntegrationError(Exception):
    """Base exception for local TypeSafe integration failures."""


class TypeSafeSchemaError(TypeSafeIntegrationError, ValueError):
    """Raised when an output schema cannot be represented by Jev questions."""


class TypeSafeExtractionError(TypeSafeIntegrationError, ValueError):
    """Raised when constrained text extraction cannot produce valid candidates."""


class TypeSafeResponseError(TypeSafeIntegrationError, RuntimeError):
    """Raised when a TypeSafe response violates the requested question contract."""


__all__ = [
    "TypeSafeExtractionError",
    "TypeSafeIntegrationError",
    "TypeSafeResponseError",
    "TypeSafeSchemaError",
]
