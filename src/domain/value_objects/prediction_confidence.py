from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class PredictionConfidence:
    """Value object representing prediction confidence levels."""

    value: float

    # Class constants for confidence levels
    HIGH_THRESHOLD: ClassVar[float] = 0.8
    MEDIUM_THRESHOLD: ClassVar[float] = 0.6
    LOW_THRESHOLD: ClassVar[float] = 0.4

    def __post_init__(self):
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Confidence value must be between 0.0 and 1.0, got {self.value}")

    @property
    def is_high(self) -> bool:
        """Check if confidence is high."""
        return self.value >= self.HIGH_THRESHOLD

    @property
    def is_medium(self) -> bool:
        """Check if confidence is medium."""
        return self.MEDIUM_THRESHOLD <= self.value < self.HIGH_THRESHOLD

    @property
    def is_low(self) -> bool:
        """Check if confidence is low."""
        return self.value < self.MEDIUM_THRESHOLD

    @property
    def level_description(self) -> str:
        """Get a string description of the confidence level."""
        if self.is_high:
            return "high"
        elif self.is_medium:
            return "medium"
        else:
            return "low"

    def percentage(self) -> float:
        """Get confidence as percentage."""
        return self.value * 100

    @classmethod
    def from_percentage(cls, percentage: float) -> 'PredictionConfidence':
        """Create confidence from percentage value."""
        return cls(percentage / 100)