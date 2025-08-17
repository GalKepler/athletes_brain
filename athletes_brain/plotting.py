from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def savefig_nice(fig: plt.Figure, filename: Path | str, *, tight: bool = True, dpi: int = 300, **savefig_kwargs: Any) -> None:
    """Save figure with optional ``tight_layout`` and consistent DPI.

    Parameters
    ----------
    fig:
        Matplotlib figure to save.
    filename:
        Destination filename. Can be a string or :class:`~pathlib.Path`.
    tight:
        Apply :meth:`~matplotlib.figure.Figure.tight_layout` before saving.
    dpi:
        Resolution in dots per inch for the saved figure.
    **savefig_kwargs:
        Additional keyword arguments forwarded to :meth:`~matplotlib.figure.Figure.savefig`.
    """
    if tight:
        fig.tight_layout()
    fig.savefig(filename, dpi=dpi, bbox_inches="tight", transparent=True, **savefig_kwargs)
