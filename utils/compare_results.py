import argparse
import csv
import numpy as np
import pandas as pd

# Calculate average and geometric mean for each column in a DataFrame and add these values back to the DataFrame
def calculate_averages_and_geometric_means(df):
    for col in df.columns:
        if col not in ['benchmark', 'task_name']:
            df.loc['average', col] = df[col].astype(float).mean()
            df.loc['geometric_mean', col] = np.exp(np.mean(np.log(df[col].astype(float))))
        else:
            df.loc['average', col] = 'average'
            df.loc['geometric_mean', col] = 'geometric_mean'
    return df

# Normalize the values in comparison_df based on baseline_df for rows with matching name in benchmark and task_name
def normalize_matching_rows(baseline_df, comparison_df):
    normalized_df = comparison_df.copy()
    for idx, row in comparison_df.iterrows():
        benchmark = row['benchmark']
        task_name = row['task_name']
        baseline_row = baseline_df[(baseline_df['benchmark'] == benchmark) & (baseline_df['task_name'] == task_name)]
        for col in comparison_df.columns:
            if col not in ['benchmark', 'task_name']:
                if not baseline_row.empty:
                    try:
                        normalized_value = float(row[col]) / float(baseline_row.iloc[0][col])
                        normalized_df.at[idx, col] = '{:.2f}'.format(normalized_value)
                    except:
                        normalized_df.at[idx, col] = np.nan
                else:
                    normalized_df.at[idx, col] = 'N/A'
    return normalized_df

def main(args):
    # Read CSV files into DataFrames
    baseline_file = args.files[0]
    dfs = {}
    for file in args.files:
        try:
            data = pd.read_csv(file)
            print(f"Successfully read {file} with shape {data.shape}")
            
            # Strip extra space around the value and sort data by benchmark and task_name
            data.columns = data.columns.str.strip()
            data = data.astype(str)
            data = data.apply(lambda x: x.str.strip(), axis=1)
            data.sort_values(by=['benchmark', 'task_name'], inplace=True, ignore_index=True)
            dfs[file] = data

            # Remove failed test cases which contain 'N/A' in any column except 'benchmark' and 'task_name' or -1.0 for avg_runtime [us]
            mask = (data.drop(columns=['benchmark', 'task_name']).map(lambda x: x == 'N/A').any(axis=1)) | (data['avg_runtime [us]'].astype(float) == -1.0)
            dfs["success_" + file] = data[~mask].reset_index(drop=True)
            print(f"After removing failed test cases, {file} has shape {dfs["success_" + file].shape}")

        except Exception as e:
            print(f"Error reading {file}: {e}")

    # Create dataframes that contain only the test cases that are present in all success files
    common_benchmarks = set(dfs["success_" + baseline_file][['benchmark', 'task_name']].itertuples(index=False, name=None))
    for file in args.files:
        if file != baseline_file:
            common_benchmarks.intersection_update(set(dfs["success_" + file][['benchmark', 'task_name']].itertuples(index=False, name=None)))
    for file in args.files:
        dfs["common_success_" + file] = dfs["success_" + file][dfs["success_" + file][['benchmark', 'task_name']].apply(tuple, axis=1).isin(common_benchmarks)].reset_index(drop=True)
        print(f"After filtering to common benchmarks that all files have success run results, {file} has shape {dfs["common_success_" + file].shape}")

    # Calculate averages and geometric means for each file
    for file in args.files:
        dfs["common_success_" + file] = calculate_averages_and_geometric_means(dfs["common_success_" + file])

    # Calculate the normalized values
    for file in args.files:
        if file != baseline_file:
            dfs["normalized_" + file] = normalize_matching_rows(dfs["common_success_" + baseline_file], dfs["common_success_" + file])
        
    # Save the results to Excel files
    with pd.ExcelWriter('comparison_results.xlsx', engine='openpyxl') as writer:
        for file in args.files:
            baseline_file_name = baseline_file.split('.')[0]
            file_name = file.split('.')[0]
            dfs[file].to_excel(writer, sheet_name=f"{file_name}", index=False)
            dfs["common_success_" + file].to_excel(writer, sheet_name=f"common_success_{file_name}", index=False)
            if file != baseline_file:
                dfs["normalized_" + file].to_excel(writer, sheet_name=f"normalized_{file_name}_vs_{baseline_file_name}", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare CSV files")
    parser.add_argument(
        '-f',
        '--files',
        nargs='+',
        required=True,
        help='List of CSV files to compare'
    )
    args = parser.parse_args()
    main(args)
