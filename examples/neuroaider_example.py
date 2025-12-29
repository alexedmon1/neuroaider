#!/usr/bin/env python3
"""
Example: Using neuroaider to create design matrices for VBM/TBSS analysis

This script demonstrates how to use neuroaider to:
1. Load participant data from CSV/TSV
2. Validate subjects against imaging data
3. Create design matrices with proper coding
4. Generate contrasts automatically
5. Save FSL-compatible files
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from neuroaider import DesignHelper

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def example_vbm_design():
    """Example: Create design for VBM analysis"""
    print("\n" + "=" * 60)
    print("EXAMPLE: VBM Design Matrix")
    print("=" * 60 + "\n")

    # Load participant data (CSV or TSV)
    helper = DesignHelper(
        participants_file='/study/participants.tsv',
        subject_column='participant_id'
    )

    # Add covariates (continuous variables)
    helper.add_covariate('age', mean_center=True, standardize=False)
    helper.add_covariate('education_years', mean_center=True, standardize=False)

    # Add factors (categorical variables)
    helper.add_categorical('sex', coding='effect')  # Effect coding (sum-to-zero)
    helper.add_categorical('group', coding='effect')  # Effect coding

    # Add contrasts to test
    helper.add_contrast('age_positive', covariate='age', direction='+')
    helper.add_contrast('age_negative', covariate='age', direction='-')
    helper.add_contrast('education_positive', covariate='education_years', direction='+')
    helper.add_contrast('sex_difference', factor='sex', level='1')  # Male vs Female
    helper.add_contrast('group_difference', factor='group', level='1')  # Patient vs Control

    # Validate subjects against imaging data
    helper.validate(
        file_pattern='/study/vbm/GM/subjects/*_GM_mni_smooth.nii.gz',
        drop_missing=True  # Only keep subjects with imaging data
    )

    # Print summary
    print(helper.summary())

    # Save design files
    output_dir = Path('/study/vbm/GM/stats')
    output_dir.mkdir(parents=True, exist_ok=True)

    helper.save(
        design_mat_file=output_dir / 'design.mat',
        design_con_file=output_dir / 'design.con',
        contrast_names_file=output_dir / 'contrast_names.txt',
        summary_file=output_dir / 'design_summary.json'
    )

    print(f"\n✓ Design files saved to: {output_dir}")


def example_tbss_design():
    """Example: Create design for TBSS analysis"""
    print("\n" + "=" * 60)
    print("EXAMPLE: TBSS Design Matrix")
    print("=" * 60 + "\n")

    # Load participant data
    helper = DesignHelper(
        participants_file='/study/participants.tsv'
    )

    # Simple design: age + group
    helper.add_covariate('age', mean_center=True)
    helper.add_categorical('group', coding='dummy', reference='0')  # 0=control (reference)

    # Contrasts
    helper.add_contrast('age_positive', covariate='age', direction='+')
    helper.add_contrast('group_patient_vs_control', factor='group', level='1')

    # Validate
    helper.validate(
        derivatives_dir='/study/derivatives',
        drop_missing=True
    )

    # Print summary
    print(helper.summary())

    # Save
    output_dir = Path('/study/tbss/stats')
    output_dir.mkdir(parents=True, exist_ok=True)

    helper.save(
        design_mat_file=output_dir / 'design.mat',
        design_con_file=output_dir / 'design.con',
        contrast_names_file=output_dir / 'contrast_names.txt',
        summary_file=output_dir / 'design_summary.json'
    )

    print(f"\n✓ Design files saved to: {output_dir}")


def example_custom_contrasts():
    """Example: Create custom contrast vectors"""
    print("\n" + "=" * 60)
    print("EXAMPLE: Custom Contrasts")
    print("=" * 60 + "\n")

    helper = DesignHelper(
        participants_file='/study/participants.tsv'
    )

    # Add variables
    helper.add_covariate('age', mean_center=True)
    helper.add_categorical('sex', coding='effect')
    helper.add_categorical('group', coding='effect')

    # Build design to see structure
    helper.build_design_matrix()
    print(f"\nDesign matrix columns: {helper.design_column_names}")

    # Custom contrast: interaction-like effect
    # [Intercept, age, sex, group]
    # Test if age effect differs between groups
    helper.add_contrast(
        name='custom_interaction',
        vector=[0, 1, 0, 1]  # Age + group
    )

    # Validate and save
    helper.validate(
        derivatives_dir='/study/derivatives',
        drop_missing=True
    )

    output_dir = Path('/study/custom')
    output_dir.mkdir(parents=True, exist_ok=True)

    helper.save(
        design_mat_file=output_dir / 'design.mat',
        design_con_file=output_dir / 'design.con',
        contrast_names_file=output_dir / 'contrast_names.txt'
    )

    print(f"\n✓ Design files saved to: {output_dir}")


def example_binary_group_comparison():
    """Example: Binary group comparison with no intercept (NEW)"""
    print("\n" + "=" * 60)
    print("EXAMPLE: Binary Group Comparison (No Intercept)")
    print("=" * 60 + "\n")

    # Load participant data
    helper = DesignHelper(
        participants_file='/study/participants.tsv',
        add_intercept=False  # KEY: No intercept for direct group mean modeling
    )

    # Add binary categorical variable (e.g., mriglu: 1=Controlled, 2=Uncontrolled)
    # Using dummy coding with no intercept creates columns for BOTH groups
    helper.add_categorical('mriglu', coding='dummy')

    # Add covariates (mean-centered)
    helper.add_covariate('age', mean_center=True)
    helper.add_covariate('sex', mean_center=False)  # Often binary 0/1, don't center

    # Automatically generate binary group contrasts
    # This creates [1, -1, 0, 0] for positive and [-1, 1, 0, 0] for negative
    helper.add_binary_group_contrasts('mriglu')

    # Validate subjects
    helper.validate(
        file_pattern='/study/vbm/GM/subjects/*_GM_mni_smooth.nii.gz',
        drop_missing=True
    )

    # Print summary
    print(helper.summary())

    # Save design files
    output_dir = Path('/study/vbm/GM/stats_binary')
    output_dir.mkdir(parents=True, exist_ok=True)

    helper.save(
        design_mat_file=output_dir / 'design.mat',
        design_con_file=output_dir / 'design.con',
        contrast_names_file=output_dir / 'contrast_names.txt',
        summary_file=output_dir / 'design_summary.json'
    )

    print(f"\n✓ Design files saved to: {output_dir}")
    print("\nDesign Matrix Columns:", helper.design_column_names)
    print("- mriglu_1: Controlled group (value=1 for controlled subjects, 0 otherwise)")
    print("- mriglu_2: Uncontrolled group (value=1 for uncontrolled subjects, 0 otherwise)")
    print("\nContrasts:")
    print("- mriglu_positive [1, -1, 0, 0]: Controlled > Uncontrolled")
    print("- mriglu_negative [-1, 1, 0, 0]: Uncontrolled > Controlled")


if __name__ == '__main__':
    # Run examples
    example_vbm_design()
    # example_tbss_design()
    # example_custom_contrasts()
    # example_binary_group_comparison()
