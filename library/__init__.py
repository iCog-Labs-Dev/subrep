"""
SubRep Skill Library — runtime storage and selection of certified skills.

Public API:
    SkillEntry    — dataclass holding one skill's metadata + policy + stats
    Certificate   — dataclass summarizing a certification gate verdict
    SkillLibrary  — in-memory store with query, save/load, and validation
    SkillSelector — pluggable selection strategies (random, payoff, MDN)
"""

from .skill_metadata import Certificate, SkillEntry
from .skill_library import SkillLibrary
from .skill_selector import SkillSelector
