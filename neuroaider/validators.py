"""
Subject validation utilities for neuroaider

Validates that subjects in participant data files have corresponding imaging data.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Set
import pandas as pd

logger = logging.getLogger(__name__)


class SubjectValidator:
    """
    Validates that subjects in participant data have corresponding imaging data

    Examples:
        # Validate against a directory of subject folders
        validator = SubjectValidator(derivatives_dir='/study/derivatives')
        valid_subjects = validator.validate(participants_df)

        # Validate against specific files
        validator = SubjectValidator(file_pattern='/study/analysis/vbm/subjects/*_GM_mni_smooth.nii.gz')
        valid_subjects = validator.validate(participants_df)
    """

    def __init__(
        self,
        derivatives_dir: Optional[Path] = None,
        file_pattern: Optional[str] = None,
        subject_column: str = 'participant_id'
    ):
        """
        Initialize subject validator

        Args:
            derivatives_dir: Directory containing subject folders (e.g., /study/derivatives)
            file_pattern: Glob pattern to find subject data files
            subject_column: Column name containing subject IDs (default: 'participant_id')

        Note: Must provide either derivatives_dir OR file_pattern
        """
        if derivatives_dir is None and file_pattern is None:
            raise ValueError("Must provide either derivatives_dir or file_pattern")

        self.derivatives_dir = Path(derivatives_dir) if derivatives_dir else None
        self.file_pattern = file_pattern
        self.subject_column = subject_column
        self.available_subjects: Optional[Set[str]] = None

    def find_available_subjects(self) -> Set[str]:
        """
        Find subjects that have imaging data

        Returns:
            Set of subject IDs with imaging data
        """
        if self.available_subjects is not None:
            return self.available_subjects

        subjects = set()

        if self.derivatives_dir:
            # Find subject directories
            if not self.derivatives_dir.exists():
                logger.warning(f"Derivatives directory not found: {self.derivatives_dir}")
                return subjects

            for subj_dir in self.derivatives_dir.iterdir():
                if subj_dir.is_dir() and not subj_dir.name.startswith('.'):
                    subjects.add(subj_dir.name)

        elif self.file_pattern:
            # Find files matching pattern and extract subject IDs
            from pathlib import Path
            pattern_path = Path(self.file_pattern)
            parent = pattern_path.parent
            pattern = pattern_path.name

            if not parent.exists():
                logger.warning(f"Pattern directory not found: {parent}")
                return subjects

            for file in parent.glob(pattern):
                # Extract subject ID from filename
                # Assumes format like: sub-001_GM_mni_smooth.nii.gz
                subject = file.name.split('_')[0]
                subjects.add(subject)

        self.available_subjects = subjects
        logger.info(f"Found {len(subjects)} subjects with imaging data")
        return subjects

    def validate(
        self,
        participants_df: pd.DataFrame,
        warn_missing: bool = True,
        drop_missing: bool = False
    ) -> pd.DataFrame:
        """
        Validate subjects in participants file against imaging data

        Args:
            participants_df: DataFrame with participant data
            warn_missing: Warn about subjects without imaging data
            drop_missing: Remove subjects without imaging data

        Returns:
            Validated DataFrame (optionally filtered)

        Raises:
            ValueError: If subject_column not in DataFrame
        """
        if self.subject_column not in participants_df.columns:
            raise ValueError(
                f"Subject column '{self.subject_column}' not found in participants file. "
                f"Available columns: {list(participants_df.columns)}"
            )

        # Find available subjects
        available = self.find_available_subjects()

        # Check each subject
        participants = set(participants_df[self.subject_column])
        missing = participants - available
        extra = available - participants
        matched = participants & available

        # Log results
        logger.info(f"Validation summary:")
        logger.info(f"  Participants in file: {len(participants)}")
        logger.info(f"  Subjects with imaging data: {len(available)}")
        logger.info(f"  Matched: {len(matched)}")

        if missing:
            msg = f"  Missing imaging data for {len(missing)} subjects: {sorted(missing)[:5]}"
            if len(missing) > 5:
                msg += f" ... and {len(missing) - 5} more"
            if warn_missing:
                logger.warning(msg)
            else:
                logger.info(msg)

        if extra:
            logger.info(f"  Imaging data without participant info: {len(extra)} subjects")

        # Optionally filter
        if drop_missing:
            df_filtered = participants_df[
                participants_df[self.subject_column].isin(available)
            ].copy()
            logger.info(f"Dropped {len(participants_df) - len(df_filtered)} subjects without imaging data")
            return df_filtered
        else:
            # Add validation column
            df_copy = participants_df.copy()
            df_copy['has_imaging_data'] = df_copy[self.subject_column].isin(available)
            return df_copy

    def get_matched_subjects(self, participants_df: pd.DataFrame) -> List[str]:
        """
        Get list of subjects that have both participant data and imaging data

        Args:
            participants_df: DataFrame with participant data

        Returns:
            List of subject IDs with both data types
        """
        available = self.find_available_subjects()
        participants = set(participants_df[self.subject_column])
        matched = sorted(participants & available)
        return matched
