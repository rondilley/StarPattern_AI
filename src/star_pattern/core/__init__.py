"""Core data types and configuration."""

from star_pattern.core.config import (
    PipelineConfig,
    DataConfig,
    DetectionConfig,
    EvolutionConfig,
    LLMConfig,
    WideFieldConfig,
    SurveyConfig,
)
from star_pattern.core.fits_handler import FITSImage
from star_pattern.core.sky_region import SkyRegion, RegionData
from star_pattern.core.catalog import CatalogEntry, StarCatalog

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
