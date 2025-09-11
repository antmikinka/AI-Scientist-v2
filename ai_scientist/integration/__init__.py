"""
Integration Module

This module provides integration components for the enhanced AI-Scientist-v2 system,
including legacy compatibility, enhanced launcher, and migration utilities.
"""

from .enhanced_launcher import EnhancedLauncher
from .legacy_adapter import LegacyAdapter
from .migration_utils import MigrationUtils

__all__ = [
    "EnhancedLauncher",
    "LegacyAdapter",
    "MigrationUtils"
]