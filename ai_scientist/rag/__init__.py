"""
Reasoning-based RAG Engine Module

This module provides advanced reasoning-based retrieval augmented generation
capabilities for AI-Scientist-v2, including PageIndex integration, LEANN reasoning,
and EmbeddingGemma processing.
"""

from .reasoning_rag_engine import ReasoningRAGEngine
from .pageindex_integration import PageIndexIntegration
from .leann_reasoner import LEANNReasoner
from .embedding_gemma_processor import EmbeddingGemmaProcessor
from .correlation_engine import CorrelationEngine

__all__ = [
    "ReasoningRAGEngine",
    "PageIndexIntegration",
    "LEANNReasoner",
    "EmbeddingGemmaProcessor",
    "CorrelationEngine"
]