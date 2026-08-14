"""Compatibility imports for pre-package synthetic-data configs.

New configs should target
``experiments.synthetic_data.gaussian.GaussianDatasetGenerator`` directly.
"""

from experiments.synthetic_data.gaussian import GaussianDatasetGenerator

DatasetGenerator = GaussianDatasetGenerator

__all__ = ["DatasetGenerator", "GaussianDatasetGenerator"]
