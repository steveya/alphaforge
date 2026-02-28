from __future__ import annotations


class PITError(Exception):
    """Base class for PIT-related errors."""


class PITContractError(PITError):
    """Raised when API contracts are violated."""


class PITValidationError(PITError):
    """Raised when PIT data fails validation."""


class PITUnsupportedOperationError(PITContractError):
    """Raised when an unsupported PIT operation is requested."""


class PITExperimentalFeatureError(PITContractError):
    """Raised when an experimental PIT feature is used without opt-in."""


class PITCausalityError(PITValidationError):
    """Raised when transform execution would violate PIT causality."""


class PITEngineError(PITError):
    """Raised when requested transform engine cannot be satisfied."""

