from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import pandas as pd


@dataclass
class DetectedPattern:
    """A base class for a detected pattern in listening history."""

    name: str
    description: str
    tracks: pd.DataFrame
    contributing_features: Dict[str, Any] = field(default_factory=dict)
    pattern_type: str = "Generic"


@dataclass
class Period(DetectedPattern):
    """Represents a significant, anomalous listening period of 2 days or more."""

    pattern_type: str = "Period"
    start_date: Any = None
    end_date: Any = None


@dataclass
class Habit(DetectedPattern):
    """Represents a recurring, cyclical listening habit."""

    pattern_type: str = "Habit"
    time_slot: Tuple = ()  # e.g., ("Wed", 8, "iPhone")
    slot_schema: str = ""  # e.g., "dow_hour", "dow_hour_platform"
