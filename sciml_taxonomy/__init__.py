"""Teleological taxonomy for Scientific Machine Learning architecture selection.

Reference implementation accompanying Wiens and Reich, "From Descriptive
Classification to Prescriptive Selection".
"""

from .cases import Case, load_cases
from .descriptor import (
    Constraints,
    DataRegime,
    Kappa,
    Knowledge,
    MacroClass,
    Task,
    Unknown,
)
from .schema import Leaf, Taxonomy
from .selection import Outcome, Result, compose, select
from .structural import analyse

__version__ = "1.0.0"

__all__ = [
    "Case",
    "Constraints",
    "DataRegime",
    "Kappa",
    "Knowledge",
    "Leaf",
    "MacroClass",
    "Outcome",
    "Result",
    "Task",
    "Taxonomy",
    "Unknown",
    "analyse",
    "compose",
    "load_cases",
    "select",
]
