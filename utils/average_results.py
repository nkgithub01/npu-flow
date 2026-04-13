import os
import re
import argparse
import csv
import math
import numpy as np
import pandas as pd
from openpyxl.styles import PatternFill, Border, Side

def main(args):
    # Check if all files exist, if not, remove them from the list
    tmp_file_list = []
    for file in args.files:
        if not os.path.isfile(file):
            print(f"Error: File {file} does not exist.")
        else:
            tmp_file_list.append(file)
    if len(tmp_file_list) == 0:
        print("Error: No valid files to process. Exiting.")
        return
    
    # Map files to labels
    args.files = tmp_file_list

    # Read CSV files into DataFrames
    dfs = {}
    for file in args.files:
        try:
            data = pd.read_csv(file)
            print(f"Successfully read {file} with shape {data.shape}")
            
            # Strip extra space around the value and sort data by benchmark and task_name with natural sorting for task_name
            data.columns = data.columns.str.strip()
            def natural_sort_key(s):
                return tuple([int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)])
            data.sort_values(by=["benchmark_name", 'task_name'], key=lambda col: col.apply(natural_sort_key), inplace=True, ignore_index=True)
            dfs[file] = data

        except Exception as e:
            print(f"Error reading {file}: {e}")
    dfs['average'] = pd.DataFrame(columns=dfs[args.files[0]].columns)  # Initialize an empty DataFrame for average results
    # Track count of contributions per cell for correct averaging
    count_matrix = pd.DataFrame(0, index=range(len(dfs[args.files[0]])), columns=dfs[args.files[0]].columns)
    
    # average results accross all benchmarks and tasks for the matching rows
    for file in args.files:
        for idx, row in dfs[file].iterrows():
            for col in dfs['average'].columns:
                if col in ["benchmark_name", "task_name"]:
                    dfs['average'].loc[idx, col] = row[col]
                else:
                    try:
                        value = float(row[col])
                        if math.isnan(value):
                            dfs['average'].loc[idx, col] = np.nan
                        else:
                            if pd.isna(dfs['average'].loc[idx, col]):
                                dfs['average'].loc[idx, col] = value
                            else:
                                dfs['average'].loc[idx, col] += value
                            count_matrix.loc[idx, col] += 1
                    except ValueError:
                        print(f"Warning: Non-numeric value '{row[col]}' in column '{col}' at row {idx} in file '{file}'. Skipping this value.")
    # Divide the summed values by the count of contributions to get the average
    num_files = len(args.files)
    for idx in dfs['average'].index:
        for col in dfs['average'].columns:
            if col not in ["benchmark_name", "task_name"]:
                try:
                    value = float(dfs['average'].loc[idx, col])
                    count = count_matrix.loc[idx, col]
                    if not math.isnan(value) and count == num_files:
                        dfs['average'].loc[idx, col] = value / count
                    else:
                        dfs['average'].loc[idx, col] = np.nan  # Set to NaN if not all files contributed to this cell
                except ValueError:
                    print(f"Warning: Non-numeric value '{dfs['average'].loc[idx, col]}' in column '{col}' at row {idx} in average results. Skipping this value.")
    
    # Save the average results to a new CSV file
    try:
        # if the output directory does not exist, create it
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        dfs['average'].to_csv(args.output_file, index=False)
        print(f"Average results saved to {args.output_file}")
    except Exception as e:
        print(f"Error saving average results to {args.output_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare CSV files")
    parser.add_argument(
        '-f',
        '--files',
        nargs='+',
        required=True,
        help='List of CSV files to compare'
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        dest='output_file',
        required=False,
        help='Output file for the comparison results',
        default='averaged_results.csv'
    )
    args = parser.parse_args()
    main(args)

