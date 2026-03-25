from dataclasses import dataclass
import re
from typing import Tuple


@dataclass(frozen=True)
class ModelVersion:
    """Value object representing a semantic version for models."""

    version: str

    def __post_init__(self):
        if not self._is_valid_version(self.version):
            raise ValueError(f"Invalid version format: {self.version}. Expected format: X.Y.Z")

    @staticmethod
    def _is_valid_version(version: str) -> bool:
        """Validate semantic version format (X.Y.Z)."""
        pattern = r"^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9\-]+))?$"
        return bool(re.match(pattern, version))

    @property
    def major(self) -> int:
        """Get the major version number."""
        return int(self.version.split('.')[0])

    @property
    def minor(self) -> int:
        """Get the minor version number."""
        return int(self.version.split('.')[1])

    @property
    def patch(self) -> int:
        """Get the patch version number."""
        patch_part = self.version.split('.')[2]
        # Handle pre-release versions (e.g., "1.2.3-beta")
        if '-' in patch_part:
            patch_part = patch_part.split('-')[0]
        return int(patch_part)

    @property
    def pre_release(self) -> str:
        """Get the pre-release identifier if present."""
        if '-' in self.version:
            return self.version.split('-', 1)[1]
        return ""

    @property
    def is_pre_release(self) -> bool:
        """Check if this is a pre-release version."""
        return bool(self.pre_release)

    def compare(self, other: 'ModelVersion') -> int:
        """
        Compare versions. Returns:
        -1 if this version is less than other
        0 if versions are equal
        1 if this version is greater than other
        """
        self_tuple = (self.major, self.minor, self.patch)
        other_tuple = (other.major, other.minor, other.patch)

        if self_tuple < other_tuple:
            return -1
        elif self_tuple > other_tuple:
            return 1
        else:
            # Same version numbers, check pre-release
            if self.is_pre_release and not other.is_pre_release:
                return -1
            elif not self.is_pre_release and other.is_pre_release:
                return 1
            elif self.is_pre_release and other.is_pre_release:
                return -1 if self.pre_release < other.pre_release else (1 if self.pre_release > other.pre_release else 0)
            else:
                return 0

    def __lt__(self, other: 'ModelVersion') -> bool:
        return self.compare(other) < 0

    def __le__(self, other: 'ModelVersion') -> bool:
        return self.compare(other) <= 0

    def __gt__(self, other: 'ModelVersion') -> bool:
        return self.compare(other) > 0

    def __ge__(self, other: 'ModelVersion') -> bool:
        return self.compare(other) >= 0

    def __eq__(self, other: 'ModelVersion') -> bool:
        return self.compare(other) == 0

    @classmethod
    def create(cls, major: int, minor: int, patch: int, pre_release: str = "") -> 'ModelVersion':
        """Create a ModelVersion from components."""
        version_str = f"{major}.{minor}.{patch}"
        if pre_release:
            version_str += f"-{pre_release}"
        return cls(version_str)