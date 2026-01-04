"""
neuroaider: Design matrix and contrast generation for neuroimaging
"""

__version__ = "0.1.0"

from .design_helper import DesignHelper
from .cli import main

__all__ = ['DesignHelper', 'main']
