"""
Baselines package for HRL Circuit Localization.

Provides implementations of:
- RandomMatchedBaseline  : uniform random edge sampling matched to circuit size k
- EAPBaseline            : Edge Attribution Patching
- EAPIGBaseline          : EAP with Integrated Gradients
- ACDCWrapper            : stub for ACDC integration
- EdgePruningWrapper     : stub for Edge-Pruning integration
"""

from .random_matched import RandomMatchedBaseline
from .eap import EAPBaseline
from .eap_ig import EAPIGBaseline
from .acdc_wrapper import ACDCWrapper
from .edge_pruning_wrapper import EdgePruningWrapper

__all__ = [
    "RandomMatchedBaseline",
    "EAPBaseline",
    "EAPIGBaseline",
    "ACDCWrapper",
    "EdgePruningWrapper",
]
