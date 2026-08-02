"""Core data types and configuration."""

from star_pattern.core.catalog import CatalogEntry, StarCatalog
from star_pattern.core.config import (
    DataConfig,
    DetectionConfig,
    EvolutionConfig,
    LLMConfig,
    PipelineConfig,
    SurveyConfig,
    WideFieldConfig,
)
from star_pattern.core.fits_handler import FITSImage
from star_pattern.core.sky_region import RegionData, SkyRegion

__all__ = [
    "PipelineConfig",
    "DataConfig",
    "DetectionConfig",
    "EvolutionConfig",
    "LLMConfig",
    "WideFieldConfig",
    "SurveyConfig",
    "FITSImage",
    "SkyRegion",
    "RegionData",
    "CatalogEntry",
    "StarCatalog",
]
