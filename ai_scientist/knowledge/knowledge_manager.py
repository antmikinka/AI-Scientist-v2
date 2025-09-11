"""
Knowledge Management System

Central knowledge management for the AI-Scientist-v2 enhanced system,
providing storage, retrieval, correlation tracking, and learning capabilities.
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import uuid
from pathlib import Path

import numpy as np


class KnowledgeType(Enum):
    THEORY = "theory"
    EXPERIMENT = "experiment"
    FINDING = "finding"
    HYPOTHESIS = "hypothesis"
    METHODOLOGY = "methodology"
    LITERATURE = "literature"
    REJECTED = "rejected"


class RelevanceLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    IRRELEVANT = "irrelevant"


@dataclass
class KnowledgeItem:
    """Represents a piece of knowledge in the system."""
    item_id: str
    title: str
    content: str
    knowledge_type: KnowledgeType
    relevance_level: RelevanceLevel
    confidence: float
    source: str
    timestamp: datetime
    
    # Metadata and relationships
    tags: List[str] = None
    related_items: List[str] = None
    correlations: Dict[str, float] = None
    context: Dict[str, Any] = None
    
    # Tracking information
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    update_count: int = 0
    last_updated: Optional[datetime] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.related_items is None:
            self.related_items = []
        if self.correlations is None:
            self.correlations = {}
        if self.context is None:
            self.context = {}


@dataclass
class RetrievalResult:
    """Result of knowledge retrieval operation."""
    query: str
    items: List[KnowledgeItem]
    relevance_scores: List[float]
    total_results: int
    retrieval_time: float
    timestamp: datetime


@dataclass
class LearningInsights:
    """Insights from learning loop analysis."""
    insight_id: str
    insights: List[str]
    patterns: List[str]
    recommendations: List[str]
    confidence: float
    timestamp: datetime


class KnowledgeManager:
    """
    Comprehensive knowledge management system for AI-Scientist-v2.
    
    Manages storage, retrieval, correlation tracking, and learning
    for all research knowledge in the system.
    """

    def __init__(self, config: Dict[str, Any], storage_backend=None, rejection_logger=None, correlation_tracker=None):
        """
        Initialize the Knowledge Manager.
        
        Args:
            config: Configuration dictionary
            storage_backend: Backend storage system
            rejection_logger: Rejection logging system
            correlation_tracker: Correlation tracking system
        """
        self.config = config
        self.storage_backend = storage_backend
        self.rejection_logger = rejection_logger
        self.correlation_tracker = correlation_tracker
        
        # Configuration parameters
        km_config = config.get("knowledge_management", {})
        self.max_knowledge_items = km_config.get("max_knowledge_items", 10000)
        self.correlation_cache_size = km_config.get("correlation_cache_size", 1000)
        self.update_frequency = km_config.get("update_frequency", "real_time")
        
        # In-memory storage (would be replaced by proper backend)
        self.knowledge_store: Dict[str, KnowledgeItem] = {}
        self.index_by_type: Dict[KnowledgeType, Set[str]] = {kt: set() for kt in KnowledgeType}
        self.index_by_tags: Dict[str, Set[str]] = {}
        self.correlation_cache: Dict[str, Dict[str, float]] = {}
        
        # Performance tracking
        self.retrieval_stats: Dict[str, Any] = {
            "total_queries": 0,
            "avg_retrieval_time": 0.0,
            "cache_hit_rate": 0.0
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)

    async def store_knowledge(self, knowledge: KnowledgeItem) -> str:
        """
        Store a knowledge item in the system.
        
        Args:
            knowledge: KnowledgeItem to store
            
        Returns:
            str: ID of stored knowledge item
        """
        if not knowledge.item_id:
            knowledge.item_id = str(uuid.uuid4())
        
        # Check storage limits
        if len(self.knowledge_store) >= self.max_knowledge_items:
            await self._cleanup_old_knowledge()
        
        # Store knowledge item
        self.knowledge_store[knowledge.item_id] = knowledge
        
        # Update indices
        self.index_by_type[knowledge.knowledge_type].add(knowledge.item_id)
        for tag in knowledge.tags:
            if tag not in self.index_by_tags:
                self.index_by_tags[tag] = set()
            self.index_by_tags[tag].add(knowledge.item_id)
        
        # Update correlations if correlation tracker is available
        if self.correlation_tracker:
            await self.correlation_tracker.add_item(knowledge)
        
        self.logger.info(f"Stored knowledge item: {knowledge.item_id} ({knowledge.knowledge_type.value})")
        
        return knowledge.item_id

    async def retrieve_knowledge(self, query: str, knowledge_types: List[KnowledgeType] = None, 
                               limit: int = 10, relevance_threshold: float = 0.0) -> RetrievalResult:
        """
        Retrieve knowledge items based on query.
        
        Args:
            query: Search query
            knowledge_types: Types of knowledge to search (None for all)
            limit: Maximum number of results
            relevance_threshold: Minimum relevance score
            
        Returns:
            RetrievalResult: Retrieved knowledge items with scores
        """
        start_time = datetime.now()
        self.retrieval_stats["total_queries"] += 1
        
        self.logger.info(f"Retrieving knowledge for query: {query}")
        
        # Filter by knowledge types if specified
        candidate_items = []
        if knowledge_types:
            for kt in knowledge_types:
                candidate_items.extend([self.knowledge_store[item_id] 
                                      for item_id in self.index_by_type[kt]])
        else:
            candidate_items = list(self.knowledge_store.values())
        
        # Score items for relevance
        scored_items = []
        for item in candidate_items:
            relevance_score = await self._calculate_relevance_score(query, item)
            if relevance_score >= relevance_threshold:
                scored_items.append((item, relevance_score))
                # Update access tracking
                item.access_count += 1
                item.last_accessed = datetime.now()
        
        # Sort by relevance and limit results
        scored_items.sort(key=lambda x: x[1], reverse=True)
        top_items = scored_items[:limit]
        
        retrieval_time = (datetime.now() - start_time).total_seconds()
        
        # Update performance stats
        self._update_retrieval_stats(retrieval_time)
        
        return RetrievalResult(
            query=query,
            items=[item for item, _ in top_items],
            relevance_scores=[score for _, score in top_items],
            total_results=len(scored_items),
            retrieval_time=retrieval_time,
            timestamp=datetime.now()
        )

    async def _calculate_relevance_score(self, query: str, item: KnowledgeItem) -> float:
        """Calculate relevance score between query and knowledge item."""
        
        # Simple text-based scoring (would be enhanced with embeddings)
        query_terms = set(query.lower().split())
        content_terms = set((item.title + " " + item.content).lower().split())
        
        # Term overlap score
        overlap = len(query_terms & content_terms)
        union = len(query_terms | content_terms)
        jaccard_score = overlap / union if union > 0 else 0.0
        
        # Type relevance boost
        type_boost = 1.0
        if item.knowledge_type in [KnowledgeType.THEORY, KnowledgeType.FINDING]:
            type_boost = 1.2
        
        # Confidence factor
        confidence_factor = item.confidence
        
        # Recency factor (more recent items get slight boost)
        recency_factor = 1.0
        if item.timestamp:
            days_old = (datetime.now() - item.timestamp).days
            recency_factor = max(0.8, 1.0 - (days_old / 365) * 0.2)
        
        return jaccard_score * type_boost * confidence_factor * recency_factor

    async def log_rejection(self, hypothesis: str, reason: str, context: Dict[str, Any] = None) -> None:
        """
        Log a rejected hypothesis or finding.
        
        Args:
            hypothesis: Rejected hypothesis text
            reason: Reason for rejection
            context: Additional context information
        """
        if self.rejection_logger:
            await self.rejection_logger.log_rejection(hypothesis, reason, context)
        
        # Also store as knowledge item
        rejected_knowledge = KnowledgeItem(
            item_id=str(uuid.uuid4()),
            title=f"Rejected: {hypothesis[:50]}...",
            content=f"Hypothesis: {hypothesis}\nReason: {reason}",
            knowledge_type=KnowledgeType.REJECTED,
            relevance_level=RelevanceLevel.LOW,
            confidence=0.0,
            source="rejection_system",
            timestamp=datetime.now(),
            context=context or {},
            tags=["rejected", "hypothesis"]
        )
        
        await self.store_knowledge(rejected_knowledge)
        
        self.logger.info(f"Logged rejection: {hypothesis[:50]}...")

    async def update_correlations(self, item_id: str, correlations: Dict[str, float]) -> None:
        """
        Update correlation information for a knowledge item.
        
        Args:
            item_id: ID of knowledge item
            correlations: Dictionary of item_id -> correlation_score
        """
        if item_id not in self.knowledge_store:
            self.logger.warning(f"Knowledge item {item_id} not found for correlation update")
            return
        
        knowledge_item = self.knowledge_store[item_id]
        knowledge_item.correlations.update(correlations)
        knowledge_item.update_count += 1
        knowledge_item.last_updated = datetime.now()
        
        # Update correlation cache
        self.correlation_cache[item_id] = correlations
        
        # Maintain cache size
        if len(self.correlation_cache) > self.correlation_cache_size:
            # Remove oldest entries
            oldest_items = sorted(self.correlation_cache.keys(), 
                                key=lambda x: self.knowledge_store[x].last_updated or datetime.min)
            for item_to_remove in oldest_items[:len(oldest_items) - self.correlation_cache_size]:
                del self.correlation_cache[item_to_remove]
        
        self.logger.info(f"Updated correlations for item {item_id}")

    async def get_learning_insights(self) -> LearningInsights:
        """
        Generate learning insights from knowledge patterns.
        
        Returns:
            LearningInsights: Patterns and recommendations
        """
        self.logger.info("Generating learning insights")
        
        insights = []
        patterns = []
        recommendations = []
        
        # Analyze knowledge distribution
        type_counts = {kt: len(items) for kt, items in self.index_by_type.items()}
        total_items = sum(type_counts.values())
        
        if total_items > 0:
            # Find dominant knowledge types
            max_type = max(type_counts, key=type_counts.get)
            max_percentage = type_counts[max_type] / total_items * 100
            
            if max_percentage > 50:
                insights.append(f"Knowledge heavily focused on {max_type.value} ({max_percentage:.1f}%)")
                recommendations.append(f"Consider diversifying knowledge collection beyond {max_type.value}")
        
        # Analyze rejection patterns
        rejected_items = self.index_by_type.get(KnowledgeType.REJECTED, set())
        if len(rejected_items) > total_items * 0.2:
            patterns.append("High rejection rate detected")
            recommendations.append("Review hypothesis generation process to reduce rejections")
        
        # Analyze access patterns
        high_access_items = [item for item in self.knowledge_store.values() if item.access_count > 5]
        if high_access_items:
            avg_confidence = sum(item.confidence for item in high_access_items) / len(high_access_items)
            if avg_confidence > 0.8:
                patterns.append("High-confidence items are frequently accessed")
                insights.append("System shows preference for high-confidence knowledge")
        
        # Analyze temporal patterns
        recent_items = [item for item in self.knowledge_store.values() 
                       if item.timestamp and (datetime.now() - item.timestamp).days < 7]
        if len(recent_items) > total_items * 0.3:
            patterns.append("High knowledge generation rate in recent period")
            insights.append("Active learning phase detected")
        
        return LearningInsights(
            insight_id=str(uuid.uuid4()),
            insights=insights or ["Insufficient data for insights"],
            patterns=patterns or ["No significant patterns detected"],
            recommendations=recommendations or ["Continue current knowledge collection"],
            confidence=0.7,
            timestamp=datetime.now()
        )

    async def find_related_knowledge(self, item_id: str, similarity_threshold: float = 0.5) -> List[Tuple[KnowledgeItem, float]]:
        """
        Find knowledge items related to a given item.
        
        Args:
            item_id: ID of reference knowledge item
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of related knowledge items with similarity scores
        """
        if item_id not in self.knowledge_store:
            return []
        
        reference_item = self.knowledge_store[item_id]
        related_items = []
        
        for other_id, other_item in self.knowledge_store.items():
            if other_id == item_id:
                continue
            
            # Calculate similarity
            similarity = await self._calculate_similarity(reference_item, other_item)
            
            if similarity >= similarity_threshold:
                related_items.append((other_item, similarity))
        
        # Sort by similarity
        related_items.sort(key=lambda x: x[1], reverse=True)
        
        return related_items

    async def _calculate_similarity(self, item1: KnowledgeItem, item2: KnowledgeItem) -> float:
        """Calculate similarity between two knowledge items."""
        
        # Tag overlap similarity
        tags1 = set(item1.tags)
        tags2 = set(item2.tags)
        tag_similarity = len(tags1 & tags2) / len(tags1 | tags2) if tags1 | tags2 else 0.0
        
        # Type similarity
        type_similarity = 1.0 if item1.knowledge_type == item2.knowledge_type else 0.3
        
        # Content similarity (simplified)
        content1_terms = set((item1.title + " " + item1.content).lower().split())
        content2_terms = set((item2.title + " " + item2.content).lower().split())
        content_similarity = len(content1_terms & content2_terms) / len(content1_terms | content2_terms) if content1_terms | content2_terms else 0.0
        
        # Weighted combination
        return (tag_similarity * 0.4 + type_similarity * 0.2 + content_similarity * 0.4)

    async def _cleanup_old_knowledge(self) -> None:
        """Clean up old or low-relevance knowledge items."""
        self.logger.info("Cleaning up old knowledge items")
        
        # Find candidates for removal
        candidates = []
        for item in self.knowledge_store.values():
            # Factors for removal: low access count, old timestamp, low confidence
            removal_score = 0.0
            
            if item.access_count == 0:
                removal_score += 0.5
            elif item.access_count < 3:
                removal_score += 0.2
            
            if item.timestamp:
                days_old = (datetime.now() - item.timestamp).days
                if days_old > 90:
                    removal_score += 0.3
            
            if item.confidence < 0.3:
                removal_score += 0.3
            
            if item.knowledge_type == KnowledgeType.REJECTED:
                removal_score += 0.4
            
            candidates.append((item, removal_score))
        
        # Sort by removal score and remove top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Remove 10% of items or minimum 100, whichever is larger
        removal_count = max(100, len(self.knowledge_store) // 10)
        
        for item, _ in candidates[:removal_count]:
            await self._remove_knowledge_item(item.item_id)

    async def _remove_knowledge_item(self, item_id: str) -> None:
        """Remove a knowledge item from all indices."""
        if item_id not in self.knowledge_store:
            return
        
        item = self.knowledge_store[item_id]
        
        # Remove from type index
        self.index_by_type[item.knowledge_type].discard(item_id)
        
        # Remove from tag indices
        for tag in item.tags:
            if tag in self.index_by_tags:
                self.index_by_tags[tag].discard(item_id)
                if not self.index_by_tags[tag]:
                    del self.index_by_tags[tag]
        
        # Remove from correlation cache
        self.correlation_cache.pop(item_id, None)
        
        # Remove from main store
        del self.knowledge_store[item_id]

    def _update_retrieval_stats(self, retrieval_time: float) -> None:
        """Update retrieval performance statistics."""
        total_queries = self.retrieval_stats["total_queries"]
        current_avg = self.retrieval_stats["avg_retrieval_time"]
        
        # Update running average
        self.retrieval_stats["avg_retrieval_time"] = (
            (current_avg * (total_queries - 1) + retrieval_time) / total_queries
        )

    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """Get comprehensive knowledge statistics."""
        total_items = len(self.knowledge_store)
        
        type_distribution = {kt.value: len(items) for kt, items in self.index_by_type.items()}
        
        avg_confidence = sum(item.confidence for item in self.knowledge_store.values()) / total_items if total_items > 0 else 0.0
        
        recent_items = sum(1 for item in self.knowledge_store.values() 
                          if item.timestamp and (datetime.now() - item.timestamp).days < 7)
        
        return {
            "total_items": total_items,
            "type_distribution": type_distribution,
            "average_confidence": avg_confidence,
            "recent_items_7_days": recent_items,
            "unique_tags": len(self.index_by_tags),
            "correlation_cache_size": len(self.correlation_cache),
            "retrieval_stats": self.retrieval_stats
        }

    def export_knowledge_base(self, export_path: str, include_rejected: bool = False) -> None:
        """Export knowledge base to file."""
        export_data = {
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "total_items": len(self.knowledge_store),
                "version": "1.0"
            },
            "knowledge_items": []
        }
        
        for item in self.knowledge_store.values():
            if not include_rejected and item.knowledge_type == KnowledgeType.REJECTED:
                continue
            
            export_data["knowledge_items"].append(asdict(item))
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Exported knowledge base to {export_path}")

    async def import_knowledge_base(self, import_path: str) -> int:
        """Import knowledge base from file."""
        try:
            with open(import_path, 'r') as f:
                import_data = json.load(f)
            
            imported_count = 0
            for item_data in import_data.get("knowledge_items", []):
                # Convert back to KnowledgeItem
                item_data["knowledge_type"] = KnowledgeType(item_data["knowledge_type"])
                item_data["relevance_level"] = RelevanceLevel(item_data["relevance_level"])
                item_data["timestamp"] = datetime.fromisoformat(item_data["timestamp"])
                
                if item_data.get("last_accessed"):
                    item_data["last_accessed"] = datetime.fromisoformat(item_data["last_accessed"])
                if item_data.get("last_updated"):
                    item_data["last_updated"] = datetime.fromisoformat(item_data["last_updated"])
                
                knowledge_item = KnowledgeItem(**item_data)
                await self.store_knowledge(knowledge_item)
                imported_count += 1
            
            self.logger.info(f"Imported {imported_count} knowledge items from {import_path}")
            return imported_count
            
        except Exception as e:
            self.logger.error(f"Error importing knowledge base: {e}")
            return 0