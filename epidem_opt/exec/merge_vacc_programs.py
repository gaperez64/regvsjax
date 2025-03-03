import argparse
from pathlib import Path

import pandas as pd

from epidem_opt.src.vacc_programs import get_all_vacc_programs_from_dir


def main():
    """
        Utility to find the minimum and maximum vaccination rates for each age group.
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--vacc_dir", type=str, required=True,
                            help="Directory for the vaccination program.")
    arg_parser.add_argument('--output_file', type=str, required=True,
                            help="Output file with the merged vaccination programs.")

    args = arg_parser.parse_args()

    vacc_dir = Path(args.vacc_dir)
    output_file = Path(args.output_file)

    if not vacc_dir.is_dir():
        print("'vacc_dir' must be an existing directory")
        exit(1)

    if output_file.is_file():
        print("Error: Output file already exists.")
        exit(1)

    vacc_programs = get_all_vacc_programs_from_dir(vacc_dir=vacc_dir)

    df_output = pd.DataFrame(vacc_programs)

    # write to csv
    df_output.to_csv(output_file, index=False)


if __name__ == '__main__':
    main()
