from pathlib import Path

import pandas as pd
import argparse

from epidem_opt.src.vacc_programs import read_min_max_rates


def main():
    """
        Utility to find the minimum and maximum vaccination rates for each age group.
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--vacc_dir", type=str, required=True,
                            help="Directory for the vaccination program.")
    arg_parser.add_argument('--output_file', type=str, required=True,
                            help="Output file with the minimum and maximum rates per age group.")

    args = arg_parser.parse_args()

    vacc_dir = Path(args.vacc_dir)
    output_file = Path(args.output_file)

    if not vacc_dir.is_dir():
        print("'vacc_dir' must be an existing directory")
        exit(1)

    if output_file.is_file():
        print("Error: Output file already exists.")
        exit(1)

    data = read_min_max_rates(vacc_dir)

    df_output = pd.DataFrame(data)

    # write to csv
    df_output.to_csv(output_file, index=False)


if __name__ == '__main__':
    main()
