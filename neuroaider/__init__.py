"""
neuroaider - Neuroimaging Analysis Design Helper

A standalone tool for creating design matrices and contrasts for neuroimaging
statistical analysis. Works with FSL, SPM, and other analysis packages.

Features:
- Load participant data from CSV/TSV files
- Validate subjects against imaging data
- Generate design matrices with proper coding
- Create contrasts automatically
- Export to FSL, SPM, or custom formats
"""

from .design_helper import DesignHelper
from .validators import SubjectValidator

__version__ = '0.1.0'
__all__ = ['DesignHelper', 'SubjectValidator']
