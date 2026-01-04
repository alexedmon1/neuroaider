
import argparse
import logging
from pathlib import Path
import sys
from neuroaider.design_helper import DesignHelper

def main():
    """Main function for the neuroaider CLI."""
    parser = argparse.ArgumentParser(
        description="""
        A command-line tool for creating design matrices and contrasts for
        neuroimaging statistical analysis, specifically for use with FSL's randomise.
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Required Arguments
    parser.add_argument('participants_file', type=Path, help='Path to the participants CSV/TSV file.')
    parser.add_argument('--output-mat', type=Path, required=True, help='Output path for the design matrix (.mat file).')
    parser.add_argument('--output-con', type=Path, required=True, help='Output path for the contrast matrix (.con file).')

    # Optional General Arguments
    parser.add_argument('--subject-column', default='participant_id', help='The name of the subject ID column in the participants file.')
    parser.add_argument('--no-intercept', action='store_true', help='Do not add an intercept to the design matrix.')

    # Covariates and Factors
    parser.add_argument('--covariate', action='append', nargs='+', metavar=('NAME', 'OPTIONS'), help='Specify a covariate. Options: mean_center=True/False, standardize=True/False.')
    parser.add_argument('--categorical', action='append', nargs='+', metavar=('NAME', 'OPTIONS'), help='Specify a categorical variable. Options: coding=effect/dummy/one-hot, reference=<level>.')

    # Contrasts
    parser.add_argument('--contrast', action='append', nargs='+', metavar='CONTRAST', help='''Add a contrast. Supported types:
    - covariate <name> <+/->
    - factor <name> <level>
    - vector <v1> <v2> ...
    ''')

    # Validation
    parser.add_argument('--validate-by-dir', type=Path, help='Directory to validate subjects against.')
    parser.add_argument('--validate-by-pattern', help='Glob pattern to validate subjects against.')
    parser.add_argument('--no-drop-missing', action='store_true', help='Do not drop subjects with missing data.')

    # Other Outputs
    parser.add_argument('--output-names', type=Path, help='Output path for contrast names (.txt file).')
    parser.add_argument('--output-summary', type=Path, help='Output path for the design summary (.json file).')

    # Logging
    parser.add_argument('-v', '--verbose', action='count', default=0, help='Increase verbosity level (e.g., -v, -vv).')

    args = parser.parse_args()

    # Set up logging
    log_level = logging.WARNING - (args.verbose * 10)
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')

    try:
        # Initialize DesignHelper
        helper = DesignHelper(args.participants_file, subject_column=args.subject_column, add_intercept=not args.no_intercept)

        # Add covariates
        if args.covariate:
            for cov_args in args.covariate:
                name = cov_args[0]
                kwargs = dict(arg.split('=') for arg in cov_args[1:])
                kwargs['mean_center'] = kwargs.get('mean_center', 'True').lower() == 'true'
                kwargs['standardize'] = kwargs.get('standardize', 'False').lower() == 'true'
                helper.add_covariate(name, **kwargs)

        # Add categoricals
        if args.categorical:
            for cat_args in args.categorical:
                name = cat_args[0]
                kwargs = dict(arg.split('=') for arg in cat_args[1:])
                helper.add_categorical(name, **kwargs)

        # Add contrasts
        if args.contrast:
            for con_args in args.contrast:
                name = con_args[0]
                con_type = con_args[1]

                if con_type == 'covariate':
                    helper.add_contrast(name, covariate=con_args[2], direction=con_args[3])
                elif con_type == 'factor':
                    helper.add_contrast(name, factor=con_args[2], level=con_args[3])
                elif con_type == 'vector':
                    helper.add_contrast(name, vector=[float(v) for v in con_args[2:]])
                else:
                    raise ValueError(f"Unknown contrast type: {con_type}")

        # Validate subjects
        if args.validate_by_dir or args.validate_by_pattern:
            helper.validate(derivatives_dir=args.validate_by_dir, file_pattern=args.validate_by_pattern, drop_missing=not args.no_drop_missing)

        # Save files
        helper.save(args.output_mat, args.output_con, contrast_names_file=args.output_names, summary_file=args.output_summary)

        print(f"Successfully generated design files:\n- {args.output_mat}\n- {args.output_con}")
        if args.output_names:
            print(f"- {args.output_names}")
        if args.output_summary:
            print(f"- {args.output_summary}")

    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
