"""
Design matrix and contrast generation for neuroimaging statistics

Simplified interface for creating FSL-compatible design matrices and contrasts
from participant data files (CSV/TSV).
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import pandas as pd
import numpy as np

from .validators import SubjectValidator

logger = logging.getLogger(__name__)


class DesignHelper:
    """
    Helper class for creating design matrices and contrasts

    Examples:
        # Load participant data
        helper = DesignHelper('participants.csv')

        # Add variables
        helper.add_covariate('age', mean_center=True)
        helper.add_categorical('sex', coding='effect', reference='F')
        helper.add_categorical('group', coding='effect', reference='control')

        # Add contrasts
        helper.add_contrast('age_positive', covariate='age', direction='+')
        helper.add_contrast('group_patient_vs_control', factor='group', level='patient')

        # Validate and save
        helper.validate(derivatives_dir='/study/derivatives')
        helper.save('design.mat', 'design.con')
    """

    def __init__(
        self,
        participants_file: Union[str, Path, pd.DataFrame],
        subject_column: str = 'participant_id',
        add_intercept: bool = True
    ):
        """
        Initialize design helper

        Args:
            participants_file: Path to CSV/TSV file or DataFrame with participant data
            subject_column: Column name containing subject IDs
            add_intercept: Include intercept column (default True). Set False for binary
                          group comparisons where you want to model group means directly.
        """
        # Load participants data
        if isinstance(participants_file, pd.DataFrame):
            self.df = participants_file.copy()
        else:
            participants_file = Path(participants_file)
            if not participants_file.exists():
                raise FileNotFoundError(f"Participants file not found: {participants_file}")

            # Auto-detect delimiter
            if participants_file.suffix == '.csv':
                self.df = pd.read_csv(participants_file)
            elif participants_file.suffix in ['.tsv', '.txt']:
                self.df = pd.read_csv(participants_file, sep='\t')
            else:
                # Try comma first, then tab
                try:
                    self.df = pd.read_csv(participants_file)
                except:
                    self.df = pd.read_csv(participants_file, sep='\t')

        self.subject_column = subject_column
        self.add_intercept = add_intercept
        self.covariates: List[Dict] = []
        self.factors: List[Dict] = []
        self.contrasts: List[Dict] = []
        self.design_matrix: Optional[np.ndarray] = None
        self.design_column_names: Optional[List[str]] = None
        self.validated = False

        logger.info(f"Loaded {len(self.df)} participants")
        logger.info(f"Available columns: {list(self.df.columns)}")

    def add_covariate(
        self,
        name: str,
        mean_center: bool = True,
        standardize: bool = False
    ):
        """
        Add continuous covariate to design

        Args:
            name: Column name in participants file
            mean_center: Center at mean (recommended for interactions)
            standardize: Z-score normalize (mean=0, std=1)
        """
        if name not in self.df.columns:
            raise ValueError(f"Column '{name}' not found in participants file")

        if not pd.api.types.is_numeric_dtype(self.df[name]):
            raise ValueError(f"Column '{name}' is not numeric")

        self.covariates.append({
            'name': name,
            'mean_center': mean_center,
            'standardize': standardize
        })
        logger.info(f"Added covariate: {name} (center={mean_center}, standardize={standardize})")

    def add_categorical(
        self,
        name: str,
        coding: str = 'effect',
        reference: Optional[str] = None
    ):
        """
        Add categorical factor to design

        Args:
            name: Column name in participants file
            coding: Coding scheme ('effect', 'dummy', or 'one-hot')
                - 'effect': Sum-to-zero (effect) coding (default, recommended for balanced designs)
                - 'dummy': Reference category = 0, others = 1 (good for unbalanced designs)
                - 'one-hot': Each level gets a column (not recommended unless you know what you're doing)
            reference: Reference category (for dummy coding)
        """
        if name not in self.df.columns:
            raise ValueError(f"Column '{name}' not found in participants file")

        if coding not in ['effect', 'dummy', 'one-hot']:
            raise ValueError(f"Coding must be 'effect', 'dummy', or 'one-hot', got '{coding}'")

        levels = sorted(self.df[name].unique())
        logger.info(f"Factor '{name}' has {len(levels)} levels: {levels}")

        if reference and reference not in levels:
            raise ValueError(f"Reference level '{reference}' not found in column '{name}'")

        self.factors.append({
            'name': name,
            'coding': coding,
            'reference': reference,
            'levels': levels
        })
        logger.info(f"Added factor: {name} (coding={coding}, reference={reference})")

    def add_contrast(
        self,
        name: str,
        covariate: Optional[str] = None,
        direction: Optional[str] = None,
        factor: Optional[str] = None,
        level: Optional[str] = None,
        vector: Optional[List[float]] = None
    ):
        """
        Add contrast to test

        Args:
            name: Contrast name
            covariate: Test covariate (provide covariate + direction)
            direction: '+' for positive, '-' for negative effect
            factor: Test factor level (provide factor + level)
            level: Level to test against reference
            vector: Custom contrast vector (advanced users)

        Examples:
            # Test positive age effect
            helper.add_contrast('age_positive', covariate='age', direction='+')

            # Test group difference
            helper.add_contrast('patient_vs_control', factor='group', level='patient')

            # Custom contrast
            helper.add_contrast('custom', vector=[0, 1, -1, 0])
        """
        if vector is not None:
            # Custom contrast vector
            self.contrasts.append({
                'name': name,
                'type': 'custom',
                'vector': vector
            })
            logger.info(f"Added custom contrast: {name}")

        elif covariate is not None:
            # Covariate contrast
            if direction not in ['+', '-']:
                raise ValueError("Direction must be '+' or '-'")

            self.contrasts.append({
                'name': name,
                'type': 'covariate',
                'covariate': covariate,
                'direction': direction
            })
            logger.info(f"Added covariate contrast: {name} ({covariate} {direction})")

        elif factor is not None and level is not None:
            # Factor level contrast
            self.contrasts.append({
                'name': name,
                'type': 'factor',
                'factor': factor,
                'level': level
            })
            logger.info(f"Added factor contrast: {name} ({factor}: {level})")

        else:
            raise ValueError(
                "Must provide either:\n"
                "  - covariate + direction\n"
                "  - factor + level\n"
                "  - vector (custom)"
            )

    def add_binary_group_contrasts(
        self,
        factor_name: str,
        positive_name: Optional[str] = None,
        negative_name: Optional[str] = None
    ):
        """
        Automatically generate contrasts for binary group comparison (no intercept mode)

        For a binary factor with levels [A, B], creates:
        - Positive contrast: [1, -1, 0, 0, ...] (A > B)
        - Negative contrast: [-1, 1, 0, 0, ...] (B > A)

        This is designed for use with add_intercept=False and dummy coding.

        Args:
            factor_name: Name of binary categorical factor
            positive_name: Name for positive contrast (default: {factor_name}_positive)
            negative_name: Name for negative contrast (default: {factor_name}_negative)

        Raises:
            ValueError: If factor not found or doesn't have exactly 2 levels

        Example:
            helper = DesignHelper('participants.tsv', add_intercept=False)
            helper.add_categorical('group', coding='dummy')  # Has levels: [1, 2]
            helper.add_covariate('age', mean_center=True)
            helper.add_binary_group_contrasts('group')
            # Creates contrasts: group_positive [1, -1, 0] and group_negative [-1, 1, 0]
        """
        # Find factor in factors list
        factor = None
        for f in self.factors:
            if f['name'] == factor_name:
                factor = f
                break

        if not factor:
            raise ValueError(
                f"Factor '{factor_name}' not found. "
                f"Add it first with add_categorical()"
            )

        if len(factor['levels']) != 2:
            raise ValueError(
                f"Binary contrast generation requires exactly 2 levels. "
                f"Factor '{factor_name}' has {len(factor['levels'])} levels: {factor['levels']}"
            )

        if not self.add_intercept and factor['coding'] != 'dummy':
            logger.warning(
                f"Binary group contrasts are typically used with dummy coding "
                f"and no intercept, but factor '{factor_name}' uses '{factor['coding']}' coding"
            )

        # Must build design matrix first to get column order
        if self.design_matrix is None:
            self.build_design_matrix()

        # Find the two group columns
        levels = sorted(factor['levels'])
        col1_name = f"{factor_name}_{levels[0]}"
        col2_name = f"{factor_name}_{levels[1]}"

        if col1_name not in self.design_column_names or col2_name not in self.design_column_names:
            raise ValueError(
                f"Could not find columns for binary groups. Expected '{col1_name}' and '{col2_name}' "
                f"but design matrix has: {self.design_column_names}"
            )

        idx1 = self.design_column_names.index(col1_name)
        idx2 = self.design_column_names.index(col2_name)

        n_predictors = len(self.design_column_names)

        # Create positive contrast: group1 > group2
        positive_vector = [0.0] * n_predictors
        positive_vector[idx1] = 1.0
        positive_vector[idx2] = -1.0

        # Create negative contrast: group2 > group1
        negative_vector = [0.0] * n_predictors
        negative_vector[idx1] = -1.0
        negative_vector[idx2] = 1.0

        # Add contrasts
        pos_name = positive_name or f"{factor_name}_positive"
        neg_name = negative_name or f"{factor_name}_negative"

        self.add_contrast(pos_name, vector=positive_vector)
        self.add_contrast(neg_name, vector=negative_vector)

        logger.info(
            f"Added binary group contrasts for '{factor_name}': "
            f"'{pos_name}' ({col1_name} > {col2_name}) and "
            f"'{neg_name}' ({col2_name} > {col1_name})"
        )

    def build_design_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Build design matrix from added covariates and factors

        Returns:
            Tuple of (design_matrix, column_names)
        """
        if len(self.covariates) == 0 and len(self.factors) == 0:
            raise ValueError("Must add at least one covariate or factor")

        columns = []
        column_names = []

        # Add intercept (optional)
        if self.add_intercept:
            columns.append(np.ones(len(self.df)))
            column_names.append('Intercept')

        # Add covariates
        for cov in self.covariates:
            values = self.df[cov['name']].values.astype(float)

            if cov['mean_center']:
                values = values - values.mean()

            if cov['standardize']:
                values = (values - values.mean()) / values.std()

            columns.append(values)
            column_names.append(cov['name'])

        # Add factors
        for factor in self.factors:
            name = factor['name']
            coding = factor['coding']
            levels = factor['levels']
            reference = factor['reference']

            if coding == 'effect':
                # Effect (sum-to-zero) coding
                # Create k-1 columns for k levels
                # Reference level coded as -1 in all columns
                if reference:
                    ref_idx = levels.index(reference)
                else:
                    ref_idx = 0
                    reference = levels[0]

                for i, level in enumerate(levels):
                    if i == ref_idx:
                        continue  # Skip reference level

                    col = np.zeros(len(self.df))
                    col[self.df[name] == level] = 1
                    col[self.df[name] == reference] = -1

                    columns.append(col)
                    column_names.append(f"{name}_{level}")

            elif coding == 'dummy':
                # Dummy coding
                # With intercept: Reference level = 0, others = 1 (k-1 columns)
                # Without intercept: Include all levels (k columns) for direct group mean modeling

                if self.add_intercept:
                    # Standard dummy coding: drop reference level
                    if reference:
                        ref_idx = levels.index(reference)
                    else:
                        ref_idx = 0
                        reference = levels[0]

                    for i, level in enumerate(levels):
                        if i == ref_idx:
                            continue  # Skip reference level

                        col = (self.df[name] == level).astype(float).values
                        columns.append(col)
                        column_names.append(f"{name}_{level}")
                else:
                    # No intercept: include ALL levels for direct group mean comparison
                    for level in levels:
                        col = (self.df[name] == level).astype(float).values
                        columns.append(col)
                        column_names.append(f"{name}_{level}")

            elif coding == 'one-hot':
                # One-hot encoding (not recommended with intercept)
                logger.warning(
                    f"One-hot encoding for {name} may cause multicollinearity "
                    "with intercept. Consider 'effect' or 'dummy' coding instead."
                )
                for level in levels:
                    col = (self.df[name] == level).astype(float).values
                    columns.append(col)
                    column_names.append(f"{name}_{level}")

        # Stack into matrix
        design_matrix = np.column_stack(columns)

        self.design_matrix = design_matrix
        self.design_column_names = column_names

        logger.info(f"Design matrix shape: {design_matrix.shape}")
        logger.info(f"Columns: {column_names}")

        return design_matrix, column_names

    def build_contrast_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Build contrast matrix from added contrasts

        Returns:
            Tuple of (contrast_matrix, contrast_names)

        Raises:
            ValueError: If design matrix not built yet or contrasts invalid
        """
        if self.design_matrix is None:
            raise ValueError("Must build design matrix first (call build_design_matrix())")

        n_predictors = self.design_matrix.shape[1]
        contrast_vectors = []
        contrast_names = []

        for contrast in self.contrasts:
            name = contrast['name']
            ctype = contrast['type']

            if ctype == 'custom':
                # Custom vector
                vector = contrast['vector']
                if len(vector) != n_predictors:
                    raise ValueError(
                        f"Contrast '{name}' has {len(vector)} values "
                        f"but design matrix has {n_predictors} predictors"
                    )

            elif ctype == 'covariate':
                # Covariate contrast
                cov_name = contrast['covariate']
                direction = contrast['direction']

                # Find column index
                if cov_name not in self.design_column_names:
                    raise ValueError(f"Covariate '{cov_name}' not found in design matrix")

                idx = self.design_column_names.index(cov_name)
                vector = [0] * n_predictors
                vector[idx] = 1 if direction == '+' else -1

            elif ctype == 'factor':
                # Factor level contrast
                factor_name = contrast['factor']
                level = contrast['level']

                # Find column for this level
                col_name = f"{factor_name}_{level}"
                if col_name not in self.design_column_names:
                    raise ValueError(
                        f"Factor level '{col_name}' not found in design matrix. "
                        f"Available columns: {self.design_column_names}"
                    )

                idx = self.design_column_names.index(col_name)
                vector = [0] * n_predictors
                vector[idx] = 1

            contrast_vectors.append(vector)
            contrast_names.append(name)

        contrast_matrix = np.array(contrast_vectors)

        logger.info(f"Contrast matrix shape: {contrast_matrix.shape}")
        logger.info(f"Contrasts: {contrast_names}")

        return contrast_matrix, contrast_names

    def validate(
        self,
        derivatives_dir: Optional[Path] = None,
        file_pattern: Optional[str] = None,
        drop_missing: bool = True
    ) -> pd.DataFrame:
        """
        Validate subjects against imaging data

        Args:
            derivatives_dir: Directory with subject data
            file_pattern: Glob pattern to find subject files
            drop_missing: Remove subjects without imaging data

        Returns:
            Validated DataFrame with matched subjects
        """
        if derivatives_dir is None and file_pattern is None:
            logger.warning("No validation performed - no imaging data location provided")
            self.validated = True
            return self.df

        validator = SubjectValidator(
            derivatives_dir=derivatives_dir,
            file_pattern=file_pattern,
            subject_column=self.subject_column
        )

        self.df = validator.validate(self.df, drop_missing=drop_missing)

        if drop_missing:
            # Rebuild design matrix with filtered subjects
            if self.design_matrix is not None:
                logger.info("Rebuilding design matrix with validated subjects")
                self.build_design_matrix()

        self.validated = True
        return self.df

    def save(
        self,
        design_mat_file: Union[str, Path],
        design_con_file: Union[str, Path],
        contrast_names_file: Optional[Union[str, Path]] = None,
        summary_file: Optional[Union[str, Path]] = None
    ):
        """
        Save design matrix and contrasts to files

        Args:
            design_mat_file: Output file for design matrix (.mat)
            design_con_file: Output file for contrasts (.con)
            contrast_names_file: Optional file for contrast names (.txt)
            summary_file: Optional JSON summary file (.json)
        """
        # Build matrices if not already done
        if self.design_matrix is None:
            self.build_design_matrix()

        design_mat, col_names = self.design_matrix, self.design_column_names
        contrast_mat, con_names = self.build_contrast_matrix()

        # Save design matrix in FSL vest format
        design_mat_file = Path(design_mat_file)
        design_mat_file.parent.mkdir(parents=True, exist_ok=True)
        with open(design_mat_file, 'w') as f:
            f.write(f"/NumWaves {design_mat.shape[1]}\n")
            f.write(f"/NumPoints {design_mat.shape[0]}\n")
            f.write("/Matrix\n")
            np.savetxt(f, design_mat, fmt='%.6f')
        logger.info(f"Saved design matrix: {design_mat_file}")

        # Save contrasts in FSL vest format
        design_con_file = Path(design_con_file)
        with open(design_con_file, 'w') as f:
            f.write(f"/NumWaves {contrast_mat.shape[1]}\n")
            f.write(f"/NumContrasts {contrast_mat.shape[0]}\n")
            f.write("/Matrix\n")
            np.savetxt(f, contrast_mat, fmt='%.6f')
        logger.info(f"Saved contrasts: {design_con_file}")

        # Save contrast names
        if contrast_names_file:
            contrast_names_file = Path(contrast_names_file)
            with open(contrast_names_file, 'w') as f:
                f.write('\n'.join(con_names))
            logger.info(f"Saved contrast names: {contrast_names_file}")

        # Save summary
        if summary_file:
            # Convert NumPy types to Python native types for JSON serialization
            def convert_to_native(obj):
                """Convert NumPy types to Python native types"""
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_native(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native(item) for item in obj]
                else:
                    return obj

            summary = {
                'n_subjects': len(self.df),
                'n_predictors': design_mat.shape[1],
                'n_contrasts': len(con_names),
                'columns': col_names,
                'contrasts': con_names,
                'covariates': convert_to_native(self.covariates),
                'factors': convert_to_native(self.factors),
                'validated': self.validated
            }

            summary_file = Path(summary_file)
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Saved summary: {summary_file}")

    def summary(self) -> str:
        """
        Get text summary of design

        Returns:
            Formatted summary string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("DESIGN SUMMARY")
        lines.append("=" * 60)
        lines.append(f"Subjects: {len(self.df)}")
        lines.append("")

        if self.covariates:
            lines.append("Covariates:")
            for cov in self.covariates:
                center = " (centered)" if cov['mean_center'] else ""
                std = " (standardized)" if cov['standardize'] else ""
                lines.append(f"  - {cov['name']}{center}{std}")
            lines.append("")

        if self.factors:
            lines.append("Factors:")
            for fac in self.factors:
                lines.append(f"  - {fac['name']} ({fac['coding']} coding)")
                lines.append(f"    Levels: {fac['levels']}")
                if fac['reference']:
                    lines.append(f"    Reference: {fac['reference']}")
            lines.append("")

        if self.design_matrix is not None:
            lines.append(f"Design Matrix: {self.design_matrix.shape}")
            lines.append(f"  Columns: {self.design_column_names}")
            lines.append("")

        if self.contrasts:
            lines.append(f"Contrasts ({len(self.contrasts)}):")
            for con in self.contrasts:
                lines.append(f"  - {con['name']}")
            lines.append("")

        lines.append(f"Validated: {self.validated}")
        lines.append("=" * 60)

        return '\n'.join(lines)
