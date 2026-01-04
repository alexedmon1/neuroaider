
import sys
from unittest.mock import patch
import pandas as pd
from neuroaider.cli import main

def test_cli_e2e(tmp_path):
    """Test the CLI end-to-end."""
    # Create a dummy participants file
    participants_file = tmp_path / "participants.csv"
    data = {'participant_id': ['sub-01', 'sub-02', 'sub-03'],
            'age': [25, 30, 35],
            'group': ['control', 'patient', 'control']}
    pd.DataFrame(data).to_csv(participants_file, index=False)

    # Define output files
    output_mat = tmp_path / "design.mat"
    output_con = tmp_path / "design.con"

    # Mock sys.argv
    with patch.object(sys, 'argv', [
        'neuroaider',
        str(participants_file),
        '--output-mat', str(output_mat),
        '--output-con', str(output_con),
        '--covariate', 'age', 'mean_center=True',
        '--categorical', 'group', 'coding=effect', 'reference=control',
        '--contrast', 'age_positive', 'covariate', 'age', '+',
        '--contrast', 'patient_vs_control', 'factor', 'group', 'patient'
    ]):
        main()

    # Check that the output files were created
    assert output_mat.exists()
    assert output_con.exists()

    # Check the content of the .mat file
    with open(output_mat, 'r') as f:
        lines = f.readlines()
    assert '/NumWaves 3' in lines[0]
    assert '/NumPoints 3' in lines[1]

    # Check the content of the .con file
    with open(output_con, 'r') as f:
        lines = f.readlines()
    assert '/NumWaves 3' in lines[0]
    assert '/NumContrasts 2' in lines[1]
