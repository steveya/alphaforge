from __future__ import annotations


class PITError(Exception):
    """Base class for PIT-related errors."""


class PITContractError(PITError):
    """Raised when API contracts are violated."""


class PITValidationError(PITError):
    """Raised when PIT data fails validation."""


class PITValidationWarning(UserWarning):
    """Warning emitted for non-blocking PIT validation outcomes."""


class PITUnsupportedOperationError(PITContractError):
    """Raised when an unsupported PIT operation is requested."""


class PITExperimentalFeatureError(PITContractError):
    """Raised when an experimental PIT feature is used without opt-in."""


class PITCausalityError(PITValidationError):
    """Raised when transform execution would violate PIT causality."""


class PITEngineError(PITError):
    """Raised when requested transform engine cannot be satisfied."""


# ---------------------------------------------------------------------------
# Adapter-level exceptions
# ---------------------------------------------------------------------------


class PITAdapterError(PITError):
    """Base exception for PIT adapter errors."""


class PITNotSupportedError(PITAdapterError):
    """Raised when a series does not support point-in-time retrieval."""

    def __init__(self, series_key: str, message: str | None = None):
        self.series_key = series_key
        if message is None:
            message = f"Series '{series_key}' does not support point-in-time retrieval"
        super().__init__(message)


class VintageNotFoundError(PITAdapterError):
    """Raised when no vintage is available for the requested asof date."""

    def __init__(self, series_key: str, asof_date: object, message: str | None = None):
        self.series_key = series_key
        self.asof_date = asof_date
        if message is None:
            message = (
                f"No vintage available for series '{series_key}' "
                f"at or before asof_date={asof_date}"
            )
        super().__init__(message)


class SourceFetchError(PITAdapterError):
    """Raised when data fetching from a source fails."""

    def __init__(
        self,
        source: str,
        message: str | None = None,
        original_error: Exception | None = None,
    ):
        self.source = source
        self.original_error = original_error
        if message is None:
            message = f"Failed to fetch data from source '{source}'"
            if original_error:
                message += f": {original_error!s}"
        super().__init__(message)
