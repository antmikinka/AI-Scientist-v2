"""
Knowledge Management Module

This module provides comprehensive knowledge management capabilities for AI-Scientist-v2,
including knowledge storage, correlation tracking, and closed-loop learning.
"""

from .knowledge_manager import KnowledgeManager
from .rejection_logger import RejectionLogger
from .correlation_tracker import CorrelationTracker
from .learning_loop import LearningLoop

__all__ = [
    "KnowledgeManager",
    "RejectionLogger",
    "CorrelationTracker", 
    "LearningLoop"
]