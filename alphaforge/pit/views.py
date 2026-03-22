"""Vintage view declarations for point-in-time data projection.

A :class:`VintageView` is a pure value object that declares *what* vintage
resolution strategy should be applied — it carries no behavior.  Pass it to a
:class:`~alphaforge.pit.resolvers.VintageResolver` to get resolution logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class VintageView:
    """Declares how the vintage dimension should be projected.

    This is a pure declaration with no behavior — it describes *what* view
    is requested, not *how* to resolve it.  Pass it to a VintageResolver to
    get resolution behavior.

    Attributes:
        mode: The vintage projection mode.
        n_releases: Only used when ``mode="frozen"``.  Which release to
            freeze to (1-indexed).  Defaults to 3, matching the common
            BEA advance → second → third GDP release sequence.
    """

    mode: Literal["realtime", "latest", "frozen"]
    n_releases: int = 3  # only used when mode="frozen"

    # Convenience constructors -------------------------------------------

    @classmethod
    def realtime(cls) -> VintageView:
        """Identity view — return values as-of the requested date."""
        return cls(mode="realtime")

    @classmethod
    def latest(cls) -> VintageView:
        """Collapse all vintages to the most recent available."""
        return cls(mode="latest")

    @classmethod
    def frozen(cls, n: int = 3) -> VintageView:
        """Freeze each observation to its *n*-th release vintage.

        Args:
            n: Release ordinal (1-indexed).  Defaults to 3.
        """
        return cls(mode="frozen", n_releases=n)
