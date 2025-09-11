"""
Theory Evolution Module

This module provides theory evolution capabilities for AI-Scientist-v2,
including theory correlation, automatic updating, and knowledge integration.
"""

from .theory_evolution_agent import TheoryEvolutionAgent
from .theory_correlator import TheoryCorrelator
from .theory_storage import TheoryStorage
from .theory_versioning import TheoryVersionManager

__all__ = [
    "TheoryEvolutionAgent",
    "TheoryCorrelator", 
    "TheoryStorage",
    "TheoryVersionManager"
]