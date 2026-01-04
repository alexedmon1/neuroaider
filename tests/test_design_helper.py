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
