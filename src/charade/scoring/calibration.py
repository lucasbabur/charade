"""Calibration maps applied to raw model probabilities (numpy only, used by serving)."""

from enum import StrEnum

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel


class CalibrationKind(StrEnum):
    """Supported maps."""

    IDENTITY = "identity"
    PLATT = "platt"
    ISOTONIC = "isotonic"


class Calibrator(BaseModel):
    """`platt`: sigmoid(a * logit + b). `isotonic`: piecewise-linear interpolation over (x, y) knots."""

    kind: CalibrationKind
    a: float = 1.0
    b: float = 0.0
    x: list[float] = []
    y: list[float] = []

    def apply(self, logits: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calibrated probabilities from raw logits, clipped to (1e-6, 1 - 1e-6)."""
        if self.kind is CalibrationKind.PLATT:
            out = 1 / (1 + np.exp(-(self.a * logits + self.b)))
        elif self.kind is CalibrationKind.ISOTONIC:
            out = np.interp(1 / (1 + np.exp(-logits)), self.x, self.y)
        else:
            out = 1 / (1 + np.exp(-logits))
        return np.clip(out, 1e-6, 1 - 1e-6)
