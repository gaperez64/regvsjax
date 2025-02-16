from pathlib import Path

import pandas as pd
import argparse


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

    # dictionaries to store min and max rates
    min_rates = {age: float("inf") for age in range(100)}
    max_rates = {age: float("-inf") for age in range(100)}
    min_files = {age: "" for age in range(100)}
    max_files = {age: "" for age in range(100)}

    # iterate over all CSVs
    # for filename in os.listdir(directory):
    for vacc_program_path in vacc_dir.glob("*.csv"):
        # read CSV
        df = pd.read_csv(vacc_program_path)

        # extract age group and vaccination rate columns
        try:
            age_col = [col for col in df.columns if 'age' in col.lower()][0]
            rate_col = [col for col in df.columns if 'rate' in col.lower()][0]
        except IndexError:
            print(f"Error when reading in vaccination program '{vacc_program_path.name}':", str(vacc_program_path))
            exit(1)

        # sanity check: every vaccination program needs to specify rates for all the age groups
        ages = set(int(age) for age in df[age_col].unique())
        assert ages == set(min_rates.keys())

        # iterate through each row to extract age and rate
        for index, row in df.iterrows():
            age = int(row[age_col])
            vaccination_rate = float(row[rate_col])

            # Obtain min rate and corresponding CSV file
            if vaccination_rate < min_rates[age]:
                min_rates[age] = vaccination_rate
                min_files[age] = vacc_program_path.name

            # Obtain max rate and corresponding CSV file
            if vaccination_rate > max_rates[age]:
                max_rates[age] = vaccination_rate
                max_files[age] = vacc_program_path.name

    # sanity check
    assert set(min_files.keys()) == set(max_files.keys()) == set(min_rates.keys()) == set(max_rates.keys())

    # create output CSV
    data = {"Age Group": list(range(100)),
            "Minimum Rate": [min_rates[age] for age in range(100)],
            "Maximum Rate": [max_rates[age] for age in range(100)],
            "CSV File Name (Minimum Rate)": [min_files[age] for age in range(100)],
            "CSV File Name (Maximum Rate)": [max_files[age] for age in range(100)]}

    df_output = pd.DataFrame(data)

    # write to csv
    df_output.to_csv(output_file, index=False)


if __name__ == '__main__':
    main()
