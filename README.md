# neuroaider

A standalone tool for creating design matrices and contrasts for neuroimaging statistical analysis.

## Features

- **CSV/TSV Support**: Load participant data from any delimited file format
- **Subject Validation**: Verify subjects have corresponding imaging data before analysis
- **Automatic Contrast Generation**: Create common contrasts without manual vector specification
- **Multiple Coding Schemes**: Effect (sum-to-zero), dummy, and one-hot coding for categorical variables
- **FSL Compatible**: Generates `.mat` and `.con` files ready for FSL randomise, TBSS, etc.
- **Type Safe**: Full validation and helpful error messages

## Quick Start

```python
from neuroaider import DesignHelper

# Load participant data
helper = DesignHelper('participants.csv')

# Add variables
helper.add_covariate('age', mean_center=True)
helper.add_categorical('sex', coding='effect')
helper.add_categorical('group', coding='effect')

# Add contrasts
helper.add_contrast('age_positive', covariate='age', direction='+')
helper.add_contrast('group_patient_vs_control', factor='group', level='patient')

# Validate subjects
helper.validate(derivatives_dir='/study/derivatives', drop_missing=True)

# Save design files
helper.save('design.mat', 'design.con', 'contrast_names.txt')
```

## Installation

```bash
# Using pip
pip install neuroaider

# Using uv
uv add neuroaider

# From source
git clone https://github.com/alexedmon1/neuroaider.git
cd neuroaider
pip install -e .
```

## Command-Line Interface (CLI)

For command-line users, `neuroaider` provides a CLI to generate design files without writing Python code. This is especially useful for quick analyses or integration into shell scripts. The CLI is intended for use with FSL's `randomise`.

**Basic Usage:**

```bash
neuroaider participants.csv \
    --output-mat design.mat \
    --output-con design.con \
    --covariate age mean_center=True \
    --categorical group coding=effect reference=control \
    --contrast age_positive covariate age + \
    --contrast patient_vs_control factor group patient
```

**Options:**

- `participants_file`: Path to your CSV or TSV file.
- `--output-mat`: Output path for the design matrix.
- `--output-con`: Output path for the contrast matrix.
- `--subject-column`: Name of the subject ID column (default: `participant_id`).
- `--no-intercept`: Disable the intercept in the design matrix.
- `--covariate`: Add a covariate (e.g., `age mean_center=True`).
- `--categorical`: Add a categorical factor (e.g., `group coding=effect reference=control`).
- `--contrast`: Add a contrast (e.g., `age_pos covariate age +`).
- `--validate-by-dir`: Validate subjects against a derivatives directory.
- `--output-names`: Save contrast names to a text file.
- `--output-summary`: Save a design summary to a JSON file.
- `-v, --verbose`: Increase output verbosity.

## Usage Examples

### VBM/TBSS Analysis

```python
from neuroaider import DesignHelper

# Load data
helper = DesignHelper('participants.tsv')

# Add covariates with mean-centering (recommended)
helper.add_covariate('age', mean_center=True)
helper.add_covariate('education_years', mean_center=True)

# Add factors with effect coding (sum-to-zero)
helper.add_categorical('sex', coding='effect')  # Reference: first level
helper.add_categorical('group', coding='effect', reference='control')

# Generate contrasts automatically
helper.add_contrast('age_positive', covariate='age', direction='+')
helper.add_contrast('age_negative', covariate='age', direction='-')
helper.add_contrast('patient_vs_control', factor='group', level='patient')

# Validate against VBM data
helper.validate(
    file_pattern='/study/vbm/subjects/*_GM_mni_smooth.nii.gz',
    drop_missing=True
)

# View summary
print(helper.summary())

# Save
helper.save('design.mat', 'design.con', summary_file='design_summary.json')
```

### Custom Contrasts

For advanced users who need specific contrast vectors:

```python
helper = DesignHelper('participants.csv')

# Add variables
helper.add_covariate('age', mean_center=True)
helper.add_categorical('group', coding='effect')

# Build design to see column order
helper.build_design_matrix()
print(f"Columns: {helper.design_column_names}")
# Output: ['Intercept', 'age', 'group_patient']

# Add custom contrast
# Test age effect in patients only
helper.add_contrast(
    name='age_in_patients',
    vector=[0, 1, 1]  # Age + group interaction
)

helper.save('design.mat', 'design.con')
```

## Coding Schemes

### Effect Coding (Recommended)

Sum-to-zero coding where reference level = -1, test level = 1:
- Good for balanced designs
- Compares each level to grand mean
- Default in most stats packages

```python
helper.add_categorical('group', coding='effect', reference='control')
# control = -1, patient = 1
```

### Dummy Coding

Reference level = 0, test level = 1:
- Good for unbalanced designs
- Compares each level to reference
- Common in regression

```python
helper.add_categorical('group', coding='dummy', reference='control')
# control = 0, patient = 1
```

### Binary Group Comparison (No Intercept)

**NEW**: For direct binary group mean comparisons without intercept:

```python
# Initialize with no intercept
helper = DesignHelper('participants.tsv', add_intercept=False)

# Add binary categorical variable (e.g., Controlled vs Uncontrolled)
helper.add_categorical('group', coding='dummy')  # levels: [1, 2]

# Add covariates
helper.add_covariate('age', mean_center=True)
helper.add_covariate('sex', mean_center=True)

# Automatically generate binary group contrasts
helper.add_binary_group_contrasts('group')
# Creates: group_positive [1, -1, 0, 0] and group_negative [-1, 1, 0, 0]
```

**Why use this?**
- Models group means directly instead of group differences
- Statistically appropriate for categorical group comparisons
- Contrasts test: Group1 > Group2 and Group2 > Group1
- Equivalent to t-test when no covariates present

**Design Matrix Structure:**
```
          group_1  group_2  age   sex
Subject1:    1       0     -5.2   0.1
Subject2:    1       0      3.1  -0.9
Subject3:    0       1     -2.1   0.1
Subject4:    0       1      4.2  -0.9
```

**Contrast Vectors:**
- `group_positive`: `[1, -1, 0, 0]` → Group1 > Group2
- `group_negative`: `[-1, 1, 0, 0]` → Group2 > Group1

### One-Hot Encoding

Each level gets its own column (not recommended with intercept):
- Creates multicollinearity issues
- Only use if you know what you're doing

## Subject Validation

Validate participants against imaging data in two ways:

### Option 1: Derivatives Directory

```python
helper.validate(
    derivatives_dir='/study/derivatives',
    drop_missing=True
)
# Looks for subject folders in derivatives directory
```

### Option 2: File Pattern

```python
helper.validate(
    file_pattern='/study/vbm/subjects/*_GM_mni_smooth.nii.gz',
    drop_missing=True
)
# Extracts subject IDs from filenames
```

## API Reference

### `DesignHelper`

Main class for creating design matrices.

**Methods:**
- `add_covariate(name, mean_center=True, standardize=False)` - Add continuous variable
- `add_categorical(name, coding='effect', reference=None)` - Add categorical variable
- `add_contrast(name, **kwargs)` - Add contrast to test
- `validate(derivatives_dir=None, file_pattern=None, drop_missing=True)` - Validate subjects
- `build_design_matrix()` - Build design matrix (called automatically by save())
- `build_contrast_matrix()` - Build contrast matrix (called automatically by save())
- `save(design_mat, design_con, contrast_names=None, summary=None)` - Save files
- `summary()` - Print design summary

### `SubjectValidator`

Validates subjects against imaging data.

**Methods:**
- `find_available_subjects()` - Find subjects with imaging data
- `validate(participants_df, warn_missing=True, drop_missing=False)` - Validate DataFrame
- `get_matched_subjects(participants_df)` - Get list of matched subjects

## Output Files

### design.mat

FSL-format design matrix (tab-separated text):
```
1.000000  -12.345678  1.000000  1.000000
1.000000  5.432109    -1.000000  1.000000
...
```

### design.con

FSL-format contrast matrix (tab-separated text):
```
0.000000  1.000000  0.000000  0.000000
0.000000  -1.000000  0.000000  0.000000
...
```

### contrast_names.txt

Human-readable contrast names (optional):
```
age_positive
age_negative
patient_vs_control
```

### design_summary.json

Complete design specification (optional):
```json
{
  "n_subjects": 23,
  "n_predictors": 4,
  "n_contrasts": 3,
  "columns": ["Intercept", "age", "sex_1", "group_1"],
  "contrasts": ["age_positive", "age_negative", "patient_vs_control"],
  ...
}
```

## Tips and Best Practices

1. **Always mean-center continuous covariates** when including categorical variables or interactions
2. **Use effect coding** for balanced designs (equal group sizes)
3. **Use dummy coding** for unbalanced designs or when you have a clear reference group
4. **Validate subjects** before running analysis to avoid mismatches
5. **Check the summary** before saving to verify your design
6. **Save the summary JSON** for reproducibility and documentation

## Integration with neurovrai

neuroaider is designed to work seamlessly with neurovrai analysis workflows:

```python
from neuroaider import DesignHelper
from neurovrai.analysis.anat.vbm_workflow import run_vbm_analysis

# Create design with neuroaider
helper = DesignHelper('participants.tsv')
helper.add_covariate('age', mean_center=True)
helper.add_categorical('group', coding='effect')
helper.add_contrast('age_positive', covariate='age', direction='+')
helper.add_contrast('group_patient_vs_control', factor='group', level='1')

helper.validate(file_pattern='/study/vbm/subjects/*_GM_mni_smooth.nii.gz')
helper.save('design.mat', 'design.con')

# Run VBM with external design
run_vbm_analysis(
    vbm_dir='/study/vbm',
    design_mat='design.mat',
    design_con='design.con',
    n_permutations=5000
)
```

## Future Plans

- CLI tool for interactive design creation
- Visualization of design matrix (heatmap)
- Support for SPM format export
- Interaction terms
- F-contrasts for multi-level factors
- Design efficiency calculations

## License

MIT License - See [LICENSE](LICENSE) for details.
