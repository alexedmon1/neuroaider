"""
Design matrix and contrast generation for neuroimaging statistics

Simplified interface for creating FSL-compatible design matrices and contrasts
from participant data files (CSV/TSV).
"""

import logging
import json
from datetime import datetime
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
        self.interactions: List[Dict] = []
        self.contrasts: List[Dict] = []
        self.ftests: List[Dict] = []
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

    def add_interaction(self, var1: str, var2: str):
        """
        Add interaction term between two variables

        Both variables must have been previously added via add_covariate() or
        add_categorical(). The interaction columns are computed as element-wise
        products of the constituent variable columns.

        Args:
            var1: Name of first variable (covariate or factor)
            var2: Name of second variable (covariate or factor)

        Examples:
            helper.add_covariate('dose', mean_center=True)
            helper.add_categorical('PND', coding='effect', reference='P30')
            helper.add_interaction('dose', 'PND')
            # Creates columns: dose×PND_P60, dose×PND_P90
        """
        cov_names = [c['name'] for c in self.covariates]
        fac_names = [f['name'] for f in self.factors]
        all_names = cov_names + fac_names

        if var1 not in all_names:
            raise ValueError(
                f"Variable '{var1}' not found. "
                f"Add it first with add_covariate() or add_categorical(). "
                f"Available: {all_names}"
            )
        if var2 not in all_names:
            raise ValueError(
                f"Variable '{var2}' not found. "
                f"Add it first with add_covariate() or add_categorical(). "
                f"Available: {all_names}"
            )
        if var1 == var2:
            raise ValueError("Cannot create interaction of a variable with itself")

        self.interactions.append({'var1': var1, 'var2': var2})
        logger.info(f"Added interaction: {var1} × {var2}")

    def add_contrast(
        self,
        name: str,
        covariate: Optional[str] = None,
        direction: Optional[str] = None,
        factor: Optional[str] = None,
        level: Optional[str] = None,
        interaction: Optional[str] = None,
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
            interaction: Interaction column name (provide interaction + level + direction)
                e.g. 'dose×PND' with level='P60' and direction='+'
            vector: Custom contrast vector (advanced users)

        Examples:
            # Test positive age effect
            helper.add_contrast('age_positive', covariate='age', direction='+')

            # Test group difference
            helper.add_contrast('patient_vs_control', factor='group', level='patient')

            # Test interaction term
            helper.add_contrast('dose_x_PND_P60', interaction='dose×PND',
                              level='P60', direction='+')

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

        elif interaction is not None:
            # Interaction contrast
            if direction not in ['+', '-']:
                raise ValueError("Direction must be '+' or '-' for interaction contrasts")
            if level is None:
                raise ValueError("Must provide level for interaction contrast")

            self.contrasts.append({
                'name': name,
                'type': 'interaction',
                'interaction': interaction,
                'level': level,
                'direction': direction
            })
            logger.info(f"Added interaction contrast: {name} ({interaction}_{level} {direction})")

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
                "  - interaction + level + direction\n"
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

    def add_ftest(
        self,
        name: str,
        contrasts: List[Union[str, int]]
    ):
        """
        Add an F-test that combines multiple t-contrasts.

        Each F-test specifies which t-contrasts (from add_contrast()) to include.
        The resulting F-test has degrees of freedom equal to the number of
        included contrasts.

        Args:
            name: F-test name (for reporting)
            contrasts: List of contrast names (str) or 1-based indices (int)
                to include in this F-test

        Examples:
            # By name
            helper.add_ftest('omnibus_dose', ['linear_pos', 'linear_neg',
                                               'quadratic_pos', 'quadratic_neg'])

            # By 1-based index (matching FSL convention)
            helper.add_ftest('deviation_from_linearity', [3, 4, 5, 6])

            # Mixed
            helper.add_ftest('linear_only', ['linear_pos', 'linear_neg'])
        """
        if not contrasts:
            raise ValueError("F-test must include at least one contrast")

        self.ftests.append({
            'name': name,
            'contrasts': contrasts
        })
        logger.info(f"Added F-test: {name} ({len(contrasts)} contrasts)")

    def build_ftest_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Build F-test specification matrix from added F-tests.

        Returns:
            Tuple of (ftest_matrix, ftest_names) where ftest_matrix has shape
            (n_ftests, n_contrasts) with 1s indicating included contrasts.

        Raises:
            ValueError: If no F-tests defined or contrast references invalid
        """
        if not self.ftests:
            raise ValueError("No F-tests defined. Use add_ftest() first.")

        if not self.contrasts:
            raise ValueError("No t-contrasts defined. F-tests require t-contrasts.")

        contrast_names = [c['name'] for c in self.contrasts]
        n_contrasts = len(contrast_names)
        n_ftests = len(self.ftests)

        ftest_matrix = np.zeros((n_ftests, n_contrasts), dtype=int)
        ftest_names = []

        for i, ftest in enumerate(self.ftests):
            ftest_names.append(ftest['name'])

            for ref in ftest['contrasts']:
                if isinstance(ref, int):
                    # 1-based index (FSL convention)
                    if ref < 1 or ref > n_contrasts:
                        raise ValueError(
                            f"F-test '{ftest['name']}': contrast index {ref} "
                            f"out of range (1-{n_contrasts})"
                        )
                    ftest_matrix[i, ref - 1] = 1
                elif isinstance(ref, str):
                    if ref not in contrast_names:
                        raise ValueError(
                            f"F-test '{ftest['name']}': contrast '{ref}' not found. "
                            f"Available: {contrast_names}"
                        )
                    ftest_matrix[i, contrast_names.index(ref)] = 1
                else:
                    raise ValueError(
                        f"F-test '{ftest['name']}': contrast reference must be "
                        f"str or int, got {type(ref)}"
                    )

        logger.info(f"F-test matrix shape: {ftest_matrix.shape}")
        logger.info(f"F-tests: {ftest_names}")

        return ftest_matrix, ftest_names

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

        # Add interaction columns
        for interaction in self.interactions:
            var1 = interaction['var1']
            var2 = interaction['var2']

            cov_names = [c['name'] for c in self.covariates]
            fac_names = [f['name'] for f in self.factors]

            var1_is_cov = var1 in cov_names
            var2_is_cov = var2 in cov_names

            if var1_is_cov and var2_is_cov:
                # Covariate × Covariate: single product column
                idx1 = column_names.index(var1)
                idx2 = column_names.index(var2)
                columns.append(columns[idx1] * columns[idx2])
                column_names.append(f"{var1}×{var2}")

            elif var1_is_cov and not var2_is_cov:
                # Covariate × Factor: one column per factor level column
                cov_idx = column_names.index(var1)
                for cn_idx, cn in enumerate(column_names):
                    if cn.startswith(f"{var2}_"):
                        level_suffix = cn[len(f"{var2}_"):]
                        columns.append(columns[cov_idx] * columns[cn_idx])
                        column_names.append(f"{var1}×{var2}_{level_suffix}")

            elif not var1_is_cov and var2_is_cov:
                # Factor × Covariate: one column per factor level column
                cov_idx = column_names.index(var2)
                for cn_idx, cn in enumerate(list(column_names)):
                    if cn.startswith(f"{var1}_"):
                        level_suffix = cn[len(f"{var1}_"):]
                        columns.append(columns[cov_idx] * columns[cn_idx])
                        column_names.append(f"{var1}_{level_suffix}×{var2}")

            else:
                # Factor × Factor: product of all column pairs
                var1_cols = [(i, cn) for i, cn in enumerate(column_names)
                             if cn.startswith(f"{var1}_")]
                var2_cols = [(i, cn) for i, cn in enumerate(column_names)
                             if cn.startswith(f"{var2}_")]
                for i1, cn1 in var1_cols:
                    l1 = cn1[len(f"{var1}_"):]
                    for i2, cn2 in var2_cols:
                        l2 = cn2[len(f"{var2}_"):]
                        columns.append(columns[i1] * columns[i2])
                        column_names.append(f"{var1}_{l1}×{var2}_{l2}")

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

            elif ctype == 'interaction':
                # Interaction contrast
                interaction_name = contrast['interaction']
                level = contrast['level']
                direction = contrast['direction']

                # Build expected column name: interaction_name + '_' + level
                col_name = f"{interaction_name}_{level}"
                if col_name not in self.design_column_names:
                    raise ValueError(
                        f"Interaction column '{col_name}' not found in design matrix. "
                        f"Available columns: {self.design_column_names}"
                    )

                idx = self.design_column_names.index(col_name)
                vector = [0] * n_predictors
                vector[idx] = 1 if direction == '+' else -1

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
        design_fts_file: Optional[Union[str, Path]] = None,
        contrast_names_file: Optional[Union[str, Path]] = None,
        summary_file: Optional[Union[str, Path]] = None
    ):
        """
        Save design matrix, contrasts, and optionally F-tests to files

        Args:
            design_mat_file: Output file for design matrix (.mat)
            design_con_file: Output file for contrasts (.con)
            design_fts_file: Optional output file for F-tests (.fts).
                Written automatically if F-tests have been added via add_ftest().
                If None but F-tests exist, writes to same directory as design_con_file.
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

        # Save F-test specification if F-tests defined
        if self.ftests:
            ftest_mat, ftest_names = self.build_ftest_matrix()

            if design_fts_file is None:
                # Default: same directory and stem as .con file
                design_fts_file = Path(design_con_file).with_suffix('.fts')

            design_fts_file = Path(design_fts_file)
            with open(design_fts_file, 'w') as f:
                f.write(f"/NumWaves {ftest_mat.shape[1]}\n")
                f.write(f"/NumContrasts {ftest_mat.shape[0]}\n")
                f.write("/Matrix\n")
                for row in ftest_mat:
                    f.write(' '.join(str(v) for v in row) + '\n')
            logger.info(f"Saved F-tests: {design_fts_file}")

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
                'n_ftests': len(self.ftests),
                'columns': col_names,
                'contrasts': con_names,
                'ftests': convert_to_native([
                    {'name': ft['name'], 'contrasts': ft['contrasts']}
                    for ft in self.ftests
                ]),
                'covariates': convert_to_native(self.covariates),
                'factors': convert_to_native(self.factors),
                'interactions': convert_to_native(self.interactions),
                'validated': self.validated
            }

            summary_file = Path(summary_file)
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Saved summary: {summary_file}")

    def write_description(self, output_path: Union[str, Path]) -> None:
        """
        Write a human-readable description of the statistical design.

        Produces a text file documenting the sample, design matrix columns
        (with coding explanations), and contrast definitions with their
        resolved vectors.

        Args:
            output_path: Path to write the description file
        """
        # Ensure design and contrast matrices are built
        if self.design_matrix is None:
            self.build_design_matrix()
        contrast_matrix, contrast_names = self.build_contrast_matrix()

        lines = []

        # Header
        lines.append("DESIGN DESCRIPTION")
        lines.append("==================")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Sample section
        lines.append("SAMPLE")
        lines.append("------")
        lines.append(f"N = {len(self.df)} subjects")

        # Group breakdown from factor columns
        factor_names = [f['name'] for f in self.factors]
        if factor_names:
            lines.append("")
            lines.append("  Group breakdown:")
            if len(factor_names) == 1:
                counts = self.df[factor_names[0]].value_counts().sort_index()
                for level, n in counts.items():
                    lines.append(f"    {factor_names[0]}={level}: {n}")
            else:
                counts = self.df.groupby(factor_names).size()
                for idx, n in counts.items():
                    if isinstance(idx, tuple):
                        label = ", ".join(
                            f"{name}={val}"
                            for name, val in zip(factor_names, idx)
                        )
                    else:
                        label = f"{factor_names[0]}={idx}"
                    lines.append(f"    {label}: {n}")

        lines.append("")

        # Design matrix section
        n_obs, n_pred = self.design_matrix.shape
        lines.append("DESIGN MATRIX")
        lines.append("-------------")
        lines.append(f"{n_obs} observations x {n_pred} predictors")
        lines.append("")

        for col_idx, col_name in enumerate(self.design_column_names):
            desc = self._describe_column(col_name, col_idx)
            lines.append(f"  Column {col_idx + 1}: {col_name} — {desc}")

        # Reference levels
        refs = [
            (f['name'], f['reference'] or f['levels'][0])
            for f in self.factors
            if f['coding'] in ('effect', 'dummy')
        ]
        if refs:
            lines.append("")
            lines.append("  Reference levels:")
            for name, ref in refs:
                lines.append(f"    {name}: {ref}")

        lines.append("")

        # Contrasts section
        lines.append("CONTRASTS")
        lines.append("---------")
        lines.append(f"{len(self.contrasts)} contrasts defined:")
        lines.append("")

        for i, (contrast, vector) in enumerate(
            zip(self.contrasts, contrast_matrix)
        ):
            vec_str = "[" + ", ".join(f"{v:g}" for v in vector) + "]"
            lines.append(f"  {i + 1}. {contrast['name']}: {vec_str}")
            explanation = self._explain_contrast(contrast, vector)
            lines.append(f"     Tests: {explanation}")
            lines.append("")

        # F-tests section
        if self.ftests:
            lines.append("F-TESTS")
            lines.append("-------")
            lines.append(f"{len(self.ftests)} F-tests defined:")
            lines.append("")

            ftest_mat, ftest_names = self.build_ftest_matrix()
            for i, (ftest, row) in enumerate(zip(self.ftests, ftest_mat)):
                included_contrasts = [
                    contrast_names[j] for j in range(len(row)) if row[j] == 1
                ]
                lines.append(f"  {i + 1}. {ftest['name']} ({len(included_contrasts)} df)")
                lines.append(f"     Includes: {', '.join(included_contrasts)}")
                vec_str = "[" + ", ".join(str(v) for v in row) + "]"
                lines.append(f"     Vector: {vec_str}")
                lines.append("")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines))
        logger.info(f"Saved design description: {output_path}")

    def _describe_column(self, col_name: str, col_idx: int) -> str:
        """Return a human-readable description of a design matrix column."""
        if col_name == "Intercept":
            return "Constant term (1 for all subjects)"

        # Check covariates
        for cov in self.covariates:
            if col_name == cov['name']:
                parts = ["Continuous"]
                if cov['standardize']:
                    parts.append("(z-scored)")
                elif cov['mean_center']:
                    parts.append("(mean-centered)")
                return " ".join(parts)

        # Check factors
        for factor in self.factors:
            prefix = f"{factor['name']}_"
            if col_name.startswith(prefix) and "×" not in col_name:
                level = col_name[len(prefix):]
                if level in [str(lv) for lv in factor['levels']]:
                    ref = factor['reference'] or factor['levels'][0]
                    if factor['coding'] == 'dummy':
                        return (
                            f"Dummy coding (ref={ref}). "
                            f"1 if {factor['name']} is {level}, 0 otherwise."
                        )
                    elif factor['coding'] == 'effect':
                        return (
                            f"Effect coding (ref={ref}). "
                            f"+1 if {factor['name']} is {level}, "
                            f"-1 if {ref}."
                        )
                    elif factor['coding'] == 'one-hot':
                        return f"One-hot. 1 if {factor['name']} is {level}, 0 otherwise."

        # Check interactions
        if "×" in col_name:
            return f"Interaction term ({col_name.replace('×', ' x ')})"

        return "Design column"

    def _explain_contrast(self, contrast: dict, vector: np.ndarray) -> str:
        """Return a human-readable explanation of what a contrast tests."""
        ctype = contrast['type']

        if ctype == 'covariate':
            cov = contrast['covariate']
            direction = "positive" if contrast['direction'] == '+' else "negative"
            return f"{direction.capitalize()} effect of {cov}"

        elif ctype == 'factor':
            factor = contrast['factor']
            level = contrast['level']
            # Find reference
            ref = None
            for f in self.factors:
                if f['name'] == factor:
                    ref = f['reference'] or f['levels'][0]
                    break
            if f and f['coding'] == 'effect':
                return f"{factor} {level} vs grand mean"
            return f"{factor} {level} vs {ref}" if ref else f"{factor} {level}"

        elif ctype == 'interaction':
            return (
                f"Interaction: {contrast['interaction']} at {contrast['level']} "
                f"({'positive' if contrast['direction'] == '+' else 'negative'})"
            )

        elif ctype == 'custom':
            # Describe based on non-zero elements
            nonzero = [
                (self.design_column_names[i], v)
                for i, v in enumerate(vector) if v != 0
            ]
            if len(nonzero) == 1:
                col, val = nonzero[0]
                direction = "positive" if val > 0 else "negative"
                return f"{direction.capitalize()} effect of {col}"
            elif len(nonzero) == 2:
                (col1, v1), (col2, v2) = nonzero
                if v1 > 0 and v2 < 0:
                    return f"{col1} > {col2}"
                elif v1 < 0 and v2 > 0:
                    return f"{col2} > {col1}"
            # Fallback: list the weights
            parts = []
            for col, val in nonzero:
                parts.append(f"{val:+g}*{col}")
            return " ".join(parts)

        return contrast['name']

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

        if self.interactions:
            lines.append("Interactions:")
            for inter in self.interactions:
                lines.append(f"  - {inter['var1']} × {inter['var2']}")
            lines.append("")

        if self.design_matrix is not None:
            lines.append(f"Design Matrix: {self.design_matrix.shape}")
            lines.append(f"  Columns: {self.design_column_names}")
            lines.append("")

        if self.contrasts:
            lines.append(f"Contrasts ({len(self.contrasts)}):")
            for i, con in enumerate(self.contrasts):
                lines.append(f"  {i+1}. {con['name']}")
            lines.append("")

        if self.ftests:
            lines.append(f"F-tests ({len(self.ftests)}):")
            contrast_names = [c['name'] for c in self.contrasts]
            for ft in self.ftests:
                included = []
                for ref in ft['contrasts']:
                    if isinstance(ref, int):
                        included.append(contrast_names[ref - 1] if ref <= len(contrast_names) else f"#{ref}")
                    else:
                        included.append(ref)
                lines.append(f"  - {ft['name']} [{', '.join(included)}]")
            lines.append("")

        lines.append(f"Validated: {self.validated}")
        lines.append("=" * 60)

        return '\n'.join(lines)
