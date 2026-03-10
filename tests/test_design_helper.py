"""Tests for DesignHelper class."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from neuroaider import DesignHelper


@pytest.fixture
def sample_participants():
    """Create sample participant data."""
    return pd.DataFrame({
        'participant_id': ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06'],
        'age': [25, 30, 35, 28, 32, 27],
        'sex': [1, 2, 1, 2, 1, 2],
        'group': [1, 1, 1, 2, 2, 2]
    })


@pytest.fixture
def helper(sample_participants):
    """Create DesignHelper with sample data."""
    return DesignHelper(sample_participants, subject_column='participant_id')


class TestDesignHelper:
    """Test DesignHelper functionality."""

    def test_init_with_dataframe(self, sample_participants):
        """Test initialization with DataFrame."""
        helper = DesignHelper(sample_participants, subject_column='participant_id')
        assert len(helper.df) == 6

    def test_add_covariate(self, helper):
        """Test adding a covariate."""
        helper.add_covariate('age', mean_center=True)
        assert 'age' in [c['name'] for c in helper.covariates]

    def test_add_categorical(self, helper):
        """Test adding a categorical variable."""
        helper.add_categorical('group', coding='effect')
        assert 'group' in [f['name'] for f in helper.factors]

    def test_build_design_matrix(self, helper):
        """Test building design matrix."""
        helper.add_covariate('age', mean_center=True)
        helper.add_categorical('group', coding='effect')
        helper.build_design_matrix()

        assert helper.design_matrix is not None
        assert helper.design_matrix.shape[0] == 6  # 6 subjects

    def test_add_contrast(self, helper):
        """Test adding contrasts."""
        helper.add_covariate('age', mean_center=True)
        helper.add_contrast('age_positive', covariate='age', direction='+')

        assert len(helper.contrasts) == 1
        assert helper.contrasts[0]['name'] == 'age_positive'

    def test_binary_group_contrasts(self, helper):
        """Test binary group contrast generation."""
        helper = DesignHelper(
            helper.df,
            subject_column='participant_id',
            add_intercept=False
        )
        helper.add_categorical('group', coding='dummy')
        helper.add_covariate('age', mean_center=True)
        helper.add_binary_group_contrasts('group')

        assert len(helper.contrasts) == 2

    def test_save_files(self, helper, tmp_path):
        """Test saving design files."""
        helper.add_covariate('age', mean_center=True)
        helper.add_categorical('group', coding='effect')
        helper.add_contrast('age_positive', covariate='age', direction='+')

        mat_file = tmp_path / 'design.mat'
        con_file = tmp_path / 'design.con'

        helper.save(
            design_mat_file=mat_file,
            design_con_file=con_file
        )

        assert mat_file.exists()
        assert con_file.exists()

    def test_summary(self, helper):
        """Test summary generation."""
        helper.add_covariate('age', mean_center=True)
        helper.add_categorical('group', coding='effect')

        summary = helper.summary()
        assert 'Subjects: 6' in summary


class TestInteractions:
    """Test interaction term functionality."""

    @pytest.fixture
    def interaction_data(self):
        """Create data suitable for interaction testing."""
        np.random.seed(42)
        n = 24
        return pd.DataFrame({
            'participant_id': [f'sub-{i:02d}' for i in range(1, n + 1)],
            'dose': np.tile([0, 1, 2, 3], n // 4),
            'PND': np.repeat(['P30', 'P60', 'P90'], n // 3),
            'sex': np.tile(['F', 'M'], n // 2),
        })

    def test_add_interaction_validates_vars(self, interaction_data):
        """Test that add_interaction validates both variables exist."""
        helper = DesignHelper(interaction_data)
        helper.add_covariate('dose', mean_center=True)

        with pytest.raises(ValueError, match="not found"):
            helper.add_interaction('dose', 'PND')

    def test_add_interaction_rejects_self(self, interaction_data):
        """Test that self-interaction is rejected."""
        helper = DesignHelper(interaction_data)
        helper.add_covariate('dose', mean_center=True)

        with pytest.raises(ValueError, match="itself"):
            helper.add_interaction('dose', 'dose')

    def test_covariate_x_factor_interaction(self, interaction_data):
        """Test covariate × factor interaction creates correct columns."""
        helper = DesignHelper(interaction_data)
        helper.add_covariate('dose', mean_center=True)
        helper.add_categorical('PND', coding='effect', reference='P30')
        helper.add_interaction('dose', 'PND')

        mat, cols = helper.build_design_matrix()

        # Should have: Intercept, dose, PND_P60, PND_P90, dose×PND_P60, dose×PND_P90
        assert 'dose×PND_P60' in cols
        assert 'dose×PND_P90' in cols
        assert mat.shape[1] == 6

        # Verify interaction values = product of constituent columns
        dose_idx = cols.index('dose')
        pnd60_idx = cols.index('PND_P60')
        int60_idx = cols.index('dose×PND_P60')
        np.testing.assert_array_almost_equal(
            mat[:, int60_idx],
            mat[:, dose_idx] * mat[:, pnd60_idx]
        )

    def test_covariate_x_covariate_interaction(self):
        """Test covariate × covariate interaction creates single column."""
        df = pd.DataFrame({
            'participant_id': [f'sub-{i}' for i in range(10)],
            'age': np.arange(10, dtype=float),
            'weight': np.arange(10, 20, dtype=float),
        })
        helper = DesignHelper(df)
        helper.add_covariate('age', mean_center=True)
        helper.add_covariate('weight', mean_center=True)
        helper.add_interaction('age', 'weight')

        mat, cols = helper.build_design_matrix()

        # Should have: Intercept, age, weight, age×weight
        assert 'age×weight' in cols
        assert mat.shape[1] == 4

    def test_factor_x_factor_interaction(self, interaction_data):
        """Test factor × factor interaction creates correct columns."""
        helper = DesignHelper(interaction_data)
        helper.add_categorical('PND', coding='effect', reference='P30')
        helper.add_categorical('sex', coding='effect', reference='F')
        helper.add_interaction('PND', 'sex')

        mat, cols = helper.build_design_matrix()

        # Should have: Intercept, PND_P60, PND_P90, sex_M, PND_P60×sex_M, PND_P90×sex_M
        assert 'PND_P60×sex_M' in cols
        assert 'PND_P90×sex_M' in cols
        assert mat.shape[1] == 6

    def test_interaction_contrast(self, interaction_data):
        """Test interaction contrast correctly targets column."""
        helper = DesignHelper(interaction_data)
        helper.add_covariate('dose', mean_center=True)
        helper.add_categorical('PND', coding='effect', reference='P30')
        helper.add_interaction('dose', 'PND')
        helper.build_design_matrix()

        helper.add_contrast('int_P60_pos', interaction='dose×PND',
                          level='P60', direction='+')
        helper.add_contrast('int_P90_neg', interaction='dose×PND',
                          level='P90', direction='-')

        con_mat, con_names = helper.build_contrast_matrix()

        cols = helper.design_column_names
        int60_idx = cols.index('dose×PND_P60')
        int90_idx = cols.index('dose×PND_P90')

        assert con_mat[0, int60_idx] == 1
        assert con_mat[1, int90_idx] == -1
        # All other entries should be 0 for these contrasts
        assert sum(abs(con_mat[0, :])) == 1
        assert sum(abs(con_mat[1, :])) == 1

    def test_interaction_in_summary(self, interaction_data):
        """Test that interactions appear in summary."""
        helper = DesignHelper(interaction_data)
        helper.add_covariate('dose', mean_center=True)
        helper.add_categorical('PND', coding='effect', reference='P30')
        helper.add_interaction('dose', 'PND')

        summary = helper.summary()
        assert 'Interactions:' in summary
        assert 'dose' in summary
        assert 'PND' in summary

    def test_full_bpa_design(self, interaction_data):
        """Test the full BPA-Rat pooled design with interaction."""
        helper = DesignHelper(interaction_data)
        helper.add_covariate('dose', mean_center=True)
        helper.add_categorical('PND', coding='effect', reference='P30')
        helper.add_categorical('sex', coding='effect', reference='F')
        helper.add_interaction('dose', 'PND')

        mat, cols = helper.build_design_matrix()

        expected_cols = [
            'Intercept', 'dose', 'PND_P60', 'PND_P90',
            'sex_M', 'dose×PND_P60', 'dose×PND_P90'
        ]
        assert cols == expected_cols
        assert mat.shape == (24, 7)

        # Add all contrasts
        helper.add_contrast('dose_pos', covariate='dose', direction='+')
        helper.add_contrast('dose_neg', covariate='dose', direction='-')
        helper.add_contrast('int_P60_pos', interaction='dose×PND',
                          level='P60', direction='+')
        helper.add_contrast('int_P60_neg', interaction='dose×PND',
                          level='P60', direction='-')
        helper.add_contrast('int_P90_pos', interaction='dose×PND',
                          level='P90', direction='+')
        helper.add_contrast('int_P90_neg', interaction='dose×PND',
                          level='P90', direction='-')

        con_mat, con_names = helper.build_contrast_matrix()
        assert con_mat.shape == (6, 7)

    def test_save_with_interactions(self, interaction_data, tmp_path):
        """Test saving design with interactions."""
        helper = DesignHelper(interaction_data)
        helper.add_covariate('dose', mean_center=True)
        helper.add_categorical('PND', coding='effect', reference='P30')
        helper.add_interaction('dose', 'PND')
        helper.add_contrast('dose_pos', covariate='dose', direction='+')
        helper.add_contrast('int_P60_pos', interaction='dose×PND',
                          level='P60', direction='+')

        mat_file = tmp_path / 'design.mat'
        con_file = tmp_path / 'design.con'
        summary_file = tmp_path / 'design_summary.json'

        helper.save(mat_file, con_file, summary_file=summary_file)

        assert mat_file.exists()
        assert con_file.exists()
        assert summary_file.exists()

        import json
        with open(summary_file) as f:
            summary = json.load(f)
        assert len(summary['interactions']) == 1
        assert summary['interactions'][0]['var1'] == 'dose'


class TestFTests:
    """Test F-test specification support."""

    @pytest.fixture
    def dose_data(self):
        """Create data with 4 dose groups for polynomial contrast testing."""
        np.random.seed(42)
        n_per_group = 5
        return pd.DataFrame({
            'participant_id': [f'sub-{i:02d}' for i in range(n_per_group * 4)],
            'dose': ['C'] * n_per_group + ['L'] * n_per_group +
                    ['M'] * n_per_group + ['H'] * n_per_group,
            'sex': (['M', 'F'] * (n_per_group * 2))[:n_per_group * 4],
        })

    def test_add_ftest_by_name(self, dose_data):
        """Test adding F-test referencing contrasts by name."""
        helper = DesignHelper(dose_data, add_intercept=False)
        helper.add_categorical('dose', coding='dummy')
        helper.build_design_matrix()

        helper.add_contrast('C_vs_L', vector=[1, 0, -1, 0])
        helper.add_contrast('C_vs_M', vector=[1, 0, 0, -1])
        helper.add_contrast('C_vs_H', vector=[1, -1, 0, 0])

        helper.add_ftest('omnibus', ['C_vs_L', 'C_vs_M', 'C_vs_H'])

        assert len(helper.ftests) == 1
        assert helper.ftests[0]['name'] == 'omnibus'

    def test_add_ftest_by_index(self, dose_data):
        """Test adding F-test referencing contrasts by 1-based index."""
        helper = DesignHelper(dose_data, add_intercept=False)
        helper.add_categorical('dose', coding='dummy')
        helper.build_design_matrix()

        helper.add_contrast('c1', vector=[1, -1, 0, 0])
        helper.add_contrast('c2', vector=[0, 0, 1, -1])

        helper.add_ftest('both', [1, 2])

        ftest_mat, ftest_names = helper.build_ftest_matrix()
        assert ftest_mat.shape == (1, 2)
        np.testing.assert_array_equal(ftest_mat[0], [1, 1])

    def test_build_ftest_matrix(self, dose_data):
        """Test that F-test matrix is correctly constructed."""
        helper = DesignHelper(dose_data, add_intercept=False)
        helper.add_categorical('dose', coding='dummy')
        helper.build_design_matrix()

        # 4 contrasts
        helper.add_contrast('c1', vector=[1, -1, 0, 0])
        helper.add_contrast('c2', vector=[0, 1, -1, 0])
        helper.add_contrast('c3', vector=[0, 0, 1, -1])
        helper.add_contrast('c4', vector=[1, 0, 0, -1])

        # F-test on first two
        helper.add_ftest('partial', ['c1', 'c2'])
        # F-test on last two
        helper.add_ftest('other', [3, 4])

        ftest_mat, ftest_names = helper.build_ftest_matrix()
        assert ftest_mat.shape == (2, 4)
        np.testing.assert_array_equal(ftest_mat[0], [1, 1, 0, 0])
        np.testing.assert_array_equal(ftest_mat[1], [0, 0, 1, 1])
        assert ftest_names == ['partial', 'other']

    def test_ftest_invalid_name_raises(self, dose_data):
        """Test that referencing a nonexistent contrast name raises."""
        helper = DesignHelper(dose_data, add_intercept=False)
        helper.add_categorical('dose', coding='dummy')
        helper.build_design_matrix()

        helper.add_contrast('c1', vector=[1, -1, 0, 0])
        helper.add_ftest('bad', ['nonexistent'])

        with pytest.raises(ValueError, match="not found"):
            helper.build_ftest_matrix()

    def test_ftest_invalid_index_raises(self, dose_data):
        """Test that out-of-range index raises."""
        helper = DesignHelper(dose_data, add_intercept=False)
        helper.add_categorical('dose', coding='dummy')
        helper.build_design_matrix()

        helper.add_contrast('c1', vector=[1, -1, 0, 0])
        helper.add_ftest('bad', [5])

        with pytest.raises(ValueError, match="out of range"):
            helper.build_ftest_matrix()

    def test_ftest_empty_raises(self, dose_data):
        """Test that empty contrast list raises."""
        helper = DesignHelper(dose_data, add_intercept=False)
        with pytest.raises(ValueError, match="at least one contrast"):
            helper.add_ftest('empty', [])

    def test_save_fts_file(self, dose_data, tmp_path):
        """Test that save() writes .fts file when F-tests defined."""
        helper = DesignHelper(dose_data, add_intercept=False)
        helper.add_categorical('dose', coding='dummy')
        helper.build_design_matrix()

        helper.add_contrast('c1', vector=[1, -1, 0, 0])
        helper.add_contrast('c2', vector=[0, 1, -1, 0])
        helper.add_contrast('c3', vector=[0, 0, 1, -1])

        helper.add_ftest('omnibus', ['c1', 'c2', 'c3'])
        helper.add_ftest('partial', ['c1', 'c2'])

        mat_file = tmp_path / 'design.mat'
        con_file = tmp_path / 'design.con'
        fts_file = tmp_path / 'design.fts'

        helper.save(mat_file, con_file, design_fts_file=fts_file)

        assert fts_file.exists()
        content = fts_file.read_text()
        assert '/NumWaves 3' in content
        assert '/NumContrasts 2' in content
        assert '1 1 1' in content   # omnibus row
        assert '1 1 0' in content   # partial row

    def test_save_auto_fts_path(self, dose_data, tmp_path):
        """Test that save() auto-generates .fts path from .con path."""
        helper = DesignHelper(dose_data, add_intercept=False)
        helper.add_categorical('dose', coding='dummy')
        helper.build_design_matrix()

        helper.add_contrast('c1', vector=[1, -1, 0, 0])
        helper.add_ftest('test', ['c1'])

        mat_file = tmp_path / 'design.mat'
        con_file = tmp_path / 'design.con'

        helper.save(mat_file, con_file)

        # Should auto-create design.fts
        assert (tmp_path / 'design.fts').exists()

    def test_no_fts_without_ftests(self, dose_data, tmp_path):
        """Test that no .fts file is written when no F-tests defined."""
        helper = DesignHelper(dose_data, add_intercept=False)
        helper.add_categorical('dose', coding='dummy')
        helper.build_design_matrix()

        helper.add_contrast('c1', vector=[1, -1, 0, 0])

        mat_file = tmp_path / 'design.mat'
        con_file = tmp_path / 'design.con'

        helper.save(mat_file, con_file)

        assert not (tmp_path / 'design.fts').exists()

    def test_summary_includes_ftests(self, dose_data):
        """Test that summary() includes F-test info."""
        helper = DesignHelper(dose_data, add_intercept=False)
        helper.add_categorical('dose', coding='dummy')
        helper.build_design_matrix()

        helper.add_contrast('c1', vector=[1, -1, 0, 0])
        helper.add_contrast('c2', vector=[0, 1, -1, 0])
        helper.add_ftest('omnibus', ['c1', 'c2'])

        summary = helper.summary()
        assert 'F-tests (1)' in summary
        assert 'omnibus' in summary

    def test_save_summary_includes_ftests(self, dose_data, tmp_path):
        """Test that JSON summary includes F-test info."""
        import json

        helper = DesignHelper(dose_data, add_intercept=False)
        helper.add_categorical('dose', coding='dummy')
        helper.build_design_matrix()

        helper.add_contrast('c1', vector=[1, -1, 0, 0])
        helper.add_ftest('test', ['c1'])

        mat_file = tmp_path / 'design.mat'
        con_file = tmp_path / 'design.con'
        summary_file = tmp_path / 'summary.json'

        helper.save(mat_file, con_file, summary_file=summary_file)

        with open(summary_file) as f:
            summary = json.load(f)

        assert summary['n_ftests'] == 1
        assert summary['ftests'][0]['name'] == 'test'
