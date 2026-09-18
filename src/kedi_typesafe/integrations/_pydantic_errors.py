from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior, UserError
from typesafe_sdk import (
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
    TypeSafeError,
)


@contextmanager
def provider_errors(model_name: str) -> Generator[None]:
    """Keep the upstream Pydantic error contract around the extended request."""
    try:
        yield
    except TypeSafeAPIResponseValidationError as exc:
        raise UnexpectedModelBehavior(
            f"Invalid response from TypeSafe: {exc}", str(exc.body)
        ) from exc
    except TypeSafeAPIError as exc:
        raise ModelHTTPError(
            status_code=exc.status, model_name=model_name, body=exc.body, headers=dict(exc.headers)
        ) from exc
    except TypeSafeAPIConnectionError as exc:
        raise ModelAPIError(model_name=model_name, message=str(exc)) from exc
    except TypeSafeError as exc:
        raise UserError(f"TypeSafe could not send this request: {exc}") from exc
