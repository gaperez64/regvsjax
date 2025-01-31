import os
import pandas as pd

# vaccination programs CSV files (current directory)
directory = "vaccination_programs"

# absolute path of the current directory
directory = os.path.abspath(directory)

# dictionaries to store min and max rates
min_rates = {age: float('inf') for age in range(100)}
max_rates = {age: float('-inf') for age in range(100)}
min_files = {age: '' for age in range(100)}
max_files = {age: '' for age in range(100)}

# iterate over all CSVs
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        
        # read CSV
        df = pd.read_csv(filepath)
        
        # extract age group and vaccination rate columns
        age_col = [col for col in df.columns if 'age' in col][0]
        rate_col = [col for col in df.columns if 'rate' in col][0]
        
        # iterate through each row to extract age and rate
        for index, row in df.iterrows():
            age = int(row[age_col])
            vaccination_rate = float(row[rate_col])
            
            # Obtain min rate and corresponding CSV file
            if vaccination_rate < min_rates[age]:
                min_rates[age] = vaccination_rate
                min_files[age] = filename
            
            # Obtain mmax rate and corresponding CSV file
            if vaccination_rate > max_rates[age]:
                max_rates[age] = vaccination_rate
                max_files[age] = filename

# create output CSV
data = {'Age Group': list(range(100)),
        'Minimum Rate': [min_rates[age] for age in range(100)],
        'Maximum Rate': [max_rates[age] for age in range(100)],
        'CSV File Name (Minimum Rate)': [min_files[age] for age in range(100)],
        'CSV File Name (Maximum Rate)': [max_files[age] for age in range(100)]}

df_output = pd.DataFrame(data)

# write to csv
output_file = "output.csv"
df_output.to_csv(output_file, index=False)
