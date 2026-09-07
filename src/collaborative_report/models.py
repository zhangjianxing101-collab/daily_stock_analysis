"""Shared data contracts for collaborative report modules."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Mapping


class ReportMode(str, Enum):
    PREMARKET = "premarket"
    POSTMARKET = "postmarket"


@dataclass(frozen=True)
class Position:
    code: str
    quantity: int
    cost_price: float


@dataclass(frozen=True)
class Candidate:
    code: str
    name: str
    horizon: str
    score: float
    close: float
    trigger: str
    stop_price: float
    target_price: float
    matched_rules: tuple[str, ...]
    observed_at: datetime
    source: str
    warning: str = ""
    industry_sector: str = ""
    concept_sectors: tuple[str, ...] = ()
    sector_rotation: str = ""
    sector_persistence: str = ""


@dataclass(frozen=True)
class ModuleResult:
    name: str
    status: str
    observed_at: datetime
    payload: Mapping[str, Any]
    warnings: tuple[str, ...] = ()
