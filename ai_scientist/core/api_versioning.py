"""
API Versioning and Lifecycle Management System

Provides comprehensive versioning, backward compatibility, and
deprecation strategies for AI-Scientist-v2 API integrations.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import json
import yaml
import warnings
from functools import wraps
import asyncio
import hashlib

logger = logging.getLogger(__name__)

class VersionStatus(Enum):
    """API version status"""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    SUNSET = "sunset"
    DEVELOPMENT = "development"
    BETA = "beta"

class CompatibilityLevel(Enum):
    """Compatibility levels"""
    FULL = "full"          # Fully compatible, no breaking changes
    BACKWARD = "backward"   # Backward compatible, new features only
    MINOR = "minor"        # Minor breaking changes, easy migration
    MAJOR = "major"        # Major breaking changes, significant migration
    INCOMPATIBLE = "incompatible"  # Incompatible, complete rewrite required

@dataclass
class Version:
    """API version information"""
    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build_metadata: Optional[str] = None

    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build_metadata:
            version += f"+{self.build_metadata}"
        return version

    def __lt__(self, other: 'Version') -> bool:
        if self.major != other.major:
            return self.major < other.major
        if self.minor != other.minor:
            return self.minor < other.minor
        return self.patch < other.patch

    def __eq__(self, other: 'Version') -> bool:
        return (self.major == other.major and
                self.minor == other.minor and
                self.patch == other.patch)

@dataclass
class VersionPolicy:
    """Versioning policy configuration"""
    support_duration: timedelta = timedelta(days=365)  # 1 year support
    deprecation_duration: timedelta = timedelta(days=90)  # 90 days deprecation notice
    sunset_duration: timedelta = timedelta(days=30)  # 30 days before sunset
    max_active_versions: int = 3
    compatibility_check: bool = True
    auto_deprecation: bool = True

@dataclass
class APILifecycleEvent:
    """API lifecycle event"""
    event_type: str
    version: Version
    timestamp: datetime
    description: str
    affected_components: List[str] = field(default_factory=list)
    migration_guide: Optional[str] = None
    breaking_changes: List[str] = field(default_factory=list)

@dataclass
class APIMigration:
    """API migration information"""
    from_version: Version
    to_version: Version
    compatibility_level: CompatibilityLevel
    migration_steps: List[str]
    breaking_changes: List[str]
    automated_migration: bool = False
    migration_script: Optional[str] = None
    estimated_effort: str = "low"  # low, medium, high

class APIVersionManager:
    """Comprehensive API versioning and lifecycle management"""

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path(__file__).parent.parent / "config" / "versioning"
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Version registry
        self._versions: Dict[str, Dict[Version, Dict[str, Any]]] = {}
        self._lifecycle_events: List[APILifecycleEvent] = []
        self._migrations: Dict[str, List[APIMigration]] = {}

        # Version policies
        self._policies: Dict[str, VersionPolicy] = {}

        # Load existing data
        self._load_versions()
        self._load_migrations()
        self._load_policies()

    def register_version(self, api_id: str, version: Version,
                       configuration: Dict[str, Any],
                       status: VersionStatus = VersionStatus.ACTIVE,
                       compatibility_level: CompatibilityLevel = CompatibilityLevel.FULL,
                       deprecation_date: Optional[datetime] = None,
                       sunset_date: Optional[datetime] = None):
        """Register a new API version"""
        if api_id not in self._versions:
            self._versions[api_id] = {}

        if version in self._versions[api_id]:
            raise ValueError(f"Version {version} already registered for {api_id}")

        # Apply versioning policy
        policy = self._policies.get(api_id, VersionPolicy())

        if deprecation_date is None and status == VersionStatus.ACTIVE:
            deprecation_date = datetime.utcnow() + policy.support_duration

        if sunset_date is None and status == VersionStatus.DEPRECATED:
            sunset_date = datetime.utcnow() + policy.sunset_duration

        version_info = {
            "version": version,
            "status": status,
            "compatibility_level": compatibility_level,
            "configuration": configuration,
            "created_at": datetime.utcnow(),
            "deprecation_date": deprecation_date,
            "sunset_date": sunset_date,
            "usage_count": 0,
            "health_score": 1.0,
            "performance_metrics": {}
        }

        self._versions[api_id][version] = version_info

        # Record lifecycle event
        self._record_lifecycle_event(
            event_type="version_registered",
            version=version,
            description=f"Registered {api_id} version {version}",
            affected_components=[api_id]
        )

        # Check version limits
        if policy.max_active_versions > 0:
            self._enforce_version_limits(api_id, policy)

        self._save_versions()
        logger.info(f"Registered {api_id} version {version}")

    def get_version(self, api_id: str, version: Version) -> Optional[Dict[str, Any]]:
        """Get specific version information"""
        return self._versions.get(api_id, {}).get(version)

    def get_latest_version(self, api_id: str, include_prerelease: bool = False) -> Optional[Version]:
        """Get the latest version of an API"""
        versions = self._versions.get(api_id, {})
        if not versions:
            return None

        # Filter out prerelease versions unless explicitly included
        active_versions = [
            v for v, info in versions.items()
            if info["status"] == VersionStatus.ACTIVE
            and (include_prerelease or v.prerelease is None)
        ]

        if not active_versions:
            return None

        return max(active_versions)

    def get_compatible_versions(self, api_id: str, current_version: Version) -> List[Version]:
        """Get compatible versions for the given version"""
        versions = self._versions.get(api_id, {})
        compatible = []

        for version, info in versions.items():
            if info["status"] == VersionStatus.SUNSET:
                continue

            compatibility = self._check_compatibility(current_version, version)
            if compatibility in [CompatibilityLevel.FULL, CompatibilityLevel.BACKWARD]:
                compatible.append(version)

        return sorted(compatible, reverse=True)

    def deprecate_version(self, api_id: str, version: Version,
                         sunset_date: Optional[datetime] = None,
                         migration_guide: Optional[str] = None):
        """Deprecate an API version"""
        version_info = self.get_version(api_id, version)
        if not version_info:
            raise ValueError(f"Version {version} not found for {api_id}")

        version_info["status"] = VersionStatus.DEPRECATED
        if sunset_date:
            version_info["sunset_date"] = sunset_date
        elif not version_info["sunset_date"]:
            policy = self._policies.get(api_id, VersionPolicy())
            version_info["sunset_date"] = datetime.utcnow() + policy.sunset_duration

        # Record lifecycle event
        self._record_lifecycle_event(
            event_type="version_deprecated",
            version=version,
            description=f"Deprecated {api_id} version {version}",
            affected_components=[api_id],
            migration_guide=migration_guide
        )

        # Issue deprecation warning
        warnings.warn(
            f"{api_id} version {version} is deprecated and will be sunset on "
            f"{version_info['sunset_date'].strftime('%Y-%m-%d')}. "
            f"Migration guide: {migration_guide}",
            DeprecationWarning,
            stacklevel=2
        )

        self._save_versions()

    def sunset_version(self, api_id: str, version: Version):
        """Sunset (retire) an API version"""
        version_info = self.get_version(api_id, version)
        if not version_info:
            raise ValueError(f"Version {version} not found for {api_id}")

        version_info["status"] = VersionStatus.SUNSET

        # Record lifecycle event
        self._record_lifecycle_event(
            event_type="version_sunset",
            version=version,
            description=f"Sunset {api_id} version {version}",
            affected_components=[api_id]
        )

        self._save_versions()
        logger.warning(f"Sunset {api_id} version {version}")

    def create_migration_plan(self, api_id: str, from_version: Version,
                           to_version: Version) -> APIMigration:
        """Create a migration plan between versions"""
        compatibility = self._check_compatibility(from_version, to_version)

        migration = APIMigration(
            from_version=from_version,
            to_version=to_version,
            compatibility_level=compatibility,
            migration_steps=self._generate_migration_steps(from_version, to_version),
            breaking_changes=self._identify_breaking_changes(from_version, to_version)
        )

        # Store migration
        if api_id not in self._migrations:
            self._migrations[api_id] = []
        self._migrations[api_id].append(migration)

        self._save_migrations()
        return migration

    def record_version_usage(self, api_id: str, version: Version,
                           request_count: int = 1,
                           performance_data: Optional[Dict[str, Any]] = None):
        """Record usage statistics for a version"""
        version_info = self.get_version(api_id, version)
        if version_info:
            version_info["usage_count"] += request_count
            if performance_data:
                version_info["performance_metrics"].update(performance_data)
            self._update_health_score(api_id, version)

    def check_version_health(self, api_id: str, version: Version) -> Dict[str, Any]:
        """Check the health status of a version"""
        version_info = self.get_version(api_id, version)
        if not version_info:
            return {"status": "not_found"}

        health_score = version_info.get("health_score", 0.0)
        usage_count = version_info.get("usage_count", 0)

        if health_score >= 0.8:
            status = "healthy"
        elif health_score >= 0.5:
            status = "degraded"
        else:
            status = "unhealthy"

        return {
            "status": status,
            "health_score": health_score,
            "usage_count": usage_count,
            "recommendations": self._generate_health_recommendations(api_id, version)
        }

    def get_version_timeline(self, api_id: str) -> List[Dict[str, Any]]:
        """Get the timeline of versions for an API"""
        versions = self._versions.get(api_id, {})
        timeline = []

        for version, info in versions.items():
            timeline.append({
                "version": str(version),
                "status": info["status"].value,
                "created_at": info["created_at"].isoformat(),
                "deprecation_date": info["deprecation_date"].isoformat() if info["deprecation_date"] else None,
                "sunset_date": info["sunset_date"].isoformat() if info["sunset_date"] else None,
                "compatibility_level": info["compatibility_level"].value,
                "usage_count": info["usage_count"],
                "health_score": info["health_score"]
            })

        return sorted(timeline, key=lambda x: x["created_at"], reverse=True)

    def check_deprecation_warnings(self) -> List[Dict[str, Any]]:
        """Check for versions that need deprecation warnings"""
        warnings = []
        now = datetime.utcnow()

        for api_id, versions in self._versions.items():
            for version, info in versions.items():
                if info["status"] == VersionStatus.ACTIVE:
                    deprecation_date = info.get("deprecation_date")
                    if deprecation_date:
                        days_until_deprecation = (deprecation_date - now).days
                        if days_until_deprecation <= 30:  # Warn 30 days before deprecation
                            warnings.append({
                                "api_id": api_id,
                                "version": str(version),
                                "days_until_deprecation": days_until_deprecation,
                                "deprecation_date": deprecation_date.isoformat(),
                                "sunset_date": info.get("sunset_date", "").isoformat() if info.get("sunset_date") else None
                            })

        return warnings

    def get_version_compatibility_matrix(self, api_id: str) -> Dict[str, Dict[str, CompatibilityLevel]]:
        """Get compatibility matrix between versions"""
        versions = self._versions.get(api_id, {})
        matrix = {}

        version_list = list(versions.keys())
        for from_version in version_list:
            matrix[str(from_version)] = {}
            for to_version in version_list:
                compatibility = self._check_compatibility(from_version, to_version)
                matrix[str(from_version)][str(to_version)] = compatibility

        return matrix

    def auto_manage_versions(self):
        """Automatically manage version lifecycle based on policies"""
        now = datetime.utcnow()
        actions_taken = []

        for api_id, versions in self._versions.items():
            policy = self._policies.get(api_id, VersionPolicy())

            if policy.auto_deprecation:
                for version, info in versions.items():
                    if info["status"] == VersionStatus.ACTIVE:
                        deprecation_date = info.get("deprecation_date")
                        if deprecation_date and now >= deprecation_date:
                            self.deprecate_version(api_id, version)
                            actions_taken.append(f"Auto-deprecated {api_id} v{version}")

                    elif info["status"] == VersionStatus.DEPRECATED:
                        sunset_date = info.get("sunset_date")
                        if sunset_date and now >= sunset_date:
                            self.sunset_version(api_id, version)
                            actions_taken.append(f"Auto-sunset {api_id} v{version}")

        return actions_taken

    def _check_compatibility(self, from_version: Version, to_version: Version) -> CompatibilityLevel:
        """Check compatibility between versions"""
        if from_version == to_version:
            return CompatibilityLevel.FULL

        # Major version change - incompatible
        if from_version.major != to_version.major:
            return CompatibilityLevel.MAJOR

        # Minor version change - backward compatible
        if from_version.minor != to_version.minor:
            return CompatibilityLevel.BACKWARD

        # Patch version change - fully compatible
        return CompatibilityLevel.FULL

    def _generate_migration_steps(self, from_version: Version, to_version: Version) -> List[str]:
        """Generate migration steps between versions"""
        compatibility = self._check_compatibility(from_version, to_version)

        if compatibility == CompatibilityLevel.FULL:
            return ["No migration required - fully compatible"]

        elif compatibility == CompatibilityLevel.BACKWARD:
            return [
                "Update version number in configuration",
                "Test with existing code",
                "Review new features documentation"
            ]

        elif compatibility == CompatibilityLevel.MAJOR:
            return [
                "Review breaking changes documentation",
                "Update API calls to match new interface",
                "Update configuration parameters",
                "Test thoroughly with staging environment",
                "Plan for potential downtime"
            ]

        else:
            return [
                "Complete rewrite may be required",
                "Consult migration guide",
                "Consider parallel running period",
                "Plan extensive testing"
            ]

    def _identify_breaking_changes(self, from_version: Version, to_version: Version) -> List[str]:
        """Identify breaking changes between versions"""
        compatibility = self._check_compatibility(from_version, to_version)
        changes = []

        if compatibility == CompatibilityLevel.MAJOR:
            changes.append("Major version change - breaking changes expected")
        elif compatibility == CompatibilityLevel.MINOR:
            changes.append("Minor version change - some breaking changes possible")

        # This would be enhanced with actual API contract comparison
        return changes

    def _update_health_score(self, api_id: str, version: Version):
        """Update health score based on usage and performance"""
        version_info = self.get_version(api_id, version)
        if not version_info:
            return

        # Simple health scoring algorithm
        usage_score = min(version_info.get("usage_count", 0) / 1000, 1.0)
        performance_score = 0.8  # Default, would be calculated from metrics
        age_score = 1.0  # Newer versions get higher scores

        health_score = (usage_score * 0.4 + performance_score * 0.4 + age_score * 0.2)
        version_info["health_score"] = health_score

    def _generate_health_recommendations(self, api_id: str, version: Version) -> List[str]:
        """Generate health improvement recommendations"""
        version_info = self.get_version(api_id, version)
        if not version_info:
            return []

        recommendations = []
        health_score = version_info.get("health_score", 0.0)

        if health_score < 0.5:
            recommendations.append("Consider deprecating this version due to low health score")

        usage_count = version_info.get("usage_count", 0)
        if usage_count < 10:
            recommendations.append("Low usage detected - consider sunsetting if not critical")

        return recommendations

    def _enforce_version_limits(self, api_id: str, policy: VersionPolicy):
        """Enforce maximum active versions policy"""
        versions = self._versions.get(api_id, {})
        active_versions = [
            (v, info) for v, info in versions.items()
            if info["status"] == VersionStatus.ACTIVE
        ]

        if len(active_versions) > policy.max_active_versions:
            # Sort by usage count and health score
            active_versions.sort(key=lambda x: (x[1].get("usage_count", 0), x[1].get("health_score", 0.0)))

            # Deprecate the oldest/least used versions
            for version, info in active_versions[:-policy.max_active_versions]:
                self.deprecate_version(api_id, version)

    def _record_lifecycle_event(self, event_type: str, version: Version,
                              description: str, affected_components: List[str],
                              migration_guide: Optional[str] = None):
        """Record a lifecycle event"""
        event = APILifecycleEvent(
            event_type=event_type,
            version=version,
            timestamp=datetime.utcnow(),
            description=description,
            affected_components=affected_components,
            migration_guide=migration_guide
        )
        self._lifecycle_events.append(event)

    def _load_versions(self):
        """Load version data from storage"""
        versions_file = self.config_dir / "versions.json"
        if versions_file.exists():
            try:
                with open(versions_file, 'r') as f:
                    data = json.load(f)

                for api_id, versions_data in data.items():
                    self._versions[api_id] = {}
                    for version_str, version_info in versions_data.items():
                        version = self._parse_version(version_str)
                        version_info["version"] = version
                        version_info["created_at"] = datetime.fromisoformat(version_info["created_at"])
                        if version_info.get("deprecation_date"):
                            version_info["deprecation_date"] = datetime.fromisoformat(version_info["deprecation_date"])
                        if version_info.get("sunset_date"):
                            version_info["sunset_date"] = datetime.fromisoformat(version_info["sunset_date"])

                        self._versions[api_id][version] = version_info
            except Exception as e:
                logger.error(f"Failed to load versions: {e}")

    def _save_versions(self):
        """Save version data to storage"""
        versions_file = self.config_dir / "versions.json"

        data = {}
        for api_id, versions in self._versions.items():
            data[api_id] = {}
            for version, info in versions.items():
                version_data = info.copy()
                version_data["version"] = str(version)
                version_data["created_at"] = info["created_at"].isoformat()
                if info.get("deprecation_date"):
                    version_data["deprecation_date"] = info["deprecation_date"].isoformat()
                if info.get("sunset_date"):
                    version_data["sunset_date"] = info["sunset_date"].isoformat()

                data[api_id][str(version)] = version_data

        try:
            with open(versions_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save versions: {e}")

    def _load_migrations(self):
        """Load migration data from storage"""
        migrations_file = self.config_dir / "migrations.json"
        if migrations_file.exists():
            try:
                with open(migrations_file, 'r') as f:
                    data = json.load(f)

                for api_id, migrations_data in data.items():
                    self._migrations[api_id] = []
                    for migration_data in migrations_data:
                        migration = APIMigration(
                            from_version=self._parse_version(migration_data["from_version"]),
                            to_version=self._parse_version(migration_data["to_version"]),
                            compatibility_level=CompatibilityLevel(migration_data["compatibility_level"]),
                            migration_steps=migration_data["migration_steps"],
                            breaking_changes=migration_data["breaking_changes"],
                            automated_migration=migration_data.get("automated_migration", False),
                            migration_script=migration_data.get("migration_script"),
                            estimated_effort=migration_data.get("estimated_effort", "low")
                        )
                        self._migrations[api_id].append(migration)
            except Exception as e:
                logger.error(f"Failed to load migrations: {e}")

    def _save_migrations(self):
        """Save migration data to storage"""
        migrations_file = self.config_dir / "migrations.json"

        data = {}
        for api_id, migrations in self._migrations.items():
            data[api_id] = []
            for migration in migrations:
                migration_data = {
                    "from_version": str(migration.from_version),
                    "to_version": str(migration.to_version),
                    "compatibility_level": migration.compatibility_level.value,
                    "migration_steps": migration.migration_steps,
                    "breaking_changes": migration.breaking_changes,
                    "automated_migration": migration.automated_migration,
                    "migration_script": migration.migration_script,
                    "estimated_effort": migration.estimated_effort
                }
                data[api_id].append(migration_data)

        try:
            with open(migrations_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save migrations: {e}")

    def _load_policies(self):
        """Load versioning policies from storage"""
        policies_file = self.config_dir / "policies.yaml"
        if policies_file.exists():
            try:
                with open(policies_file, 'r') as f:
                    data = yaml.safe_load(f)

                for api_id, policy_data in data.items():
                    self._policies[api_id] = VersionPolicy(
                        support_duration=timedelta(days=policy_data.get("support_duration", 365)),
                        deprecation_duration=timedelta(days=policy_data.get("deprecation_duration", 90)),
                        sunset_duration=timedelta(days=policy_data.get("sunset_duration", 30)),
                        max_active_versions=policy_data.get("max_active_versions", 3),
                        compatibility_check=policy_data.get("compatibility_check", True),
                        auto_deprecation=policy_data.get("auto_deprecation", True)
                    )
            except Exception as e:
                logger.error(f"Failed to load policies: {e}")

    def _parse_version(self, version_str: str) -> Version:
        """Parse version string into Version object"""
        parts = version_str.split('-')
        version_part = parts[0]
        prerelease = parts[1] if len(parts) > 1 else None

        # Split build metadata
        if prerelease and '+' in prerelease:
            prerelease, build_metadata = prerelease.split('+', 1)
        else:
            build_metadata = None

        # Parse version numbers
        version_numbers = version_part.split('.')
        if len(version_numbers) < 3:
            raise ValueError(f"Invalid version format: {version_str}")

        major = int(version_numbers[0])
        minor = int(version_numbers[1])
        patch = int(version_numbers[2])

        return Version(
            major=major,
            minor=minor,
            patch=patch,
            prerelease=prerelease,
            build_metadata=build_metadata
        )

# Decorators for version management
def version_compatible(min_version: str, max_version: str = None):
    """Decorator to check API version compatibility"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This would check the current API version against constraints
            # For now, just pass through
            return func(*args, **kwargs)
        return wrapper
    return decorator

def deprecated_version(reason: str, version: str, removal_version: str):
    """Decorator to mark functions as deprecated"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated since version {version} "
                f"and will be removed in version {removal_version}. {reason}",
                DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Global version manager instance
_version_manager = None

def get_version_manager() -> APIVersionManager:
    """Get the global version manager instance"""
    global _version_manager
    if _version_manager is None:
        _version_manager = APIVersionManager()
    return _version_manager

# Convenience functions
def register_api_version(api_id: str, version: Version, **kwargs):
    """Register a new API version"""
    manager = get_version_manager()
    return manager.register_version(api_id, version, **kwargs)

def get_latest_api_version(api_id: str) -> Optional[Version]:
    """Get the latest version of an API"""
    manager = get_version_manager()
    return manager.get_latest_version(api_id)

def check_api_version_compatibility(api_id: str, current_version: Version) -> List[Version]:
    """Get compatible versions for the given version"""
    manager = get_version_manager()
    return manager.get_compatible_versions(api_id, current_version)