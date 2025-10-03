import argparse
import csv
import numpy as np
import pandas as pd

# Calculate average and geometric mean for each column in a DataFrame and add these values back to the DataFrame
def calculate_averages_and_geometric_means(df, prefix=""):
    output_df = df.copy()
    for col in output_df.columns:
        if col not in ['benchmark', 'task_name']:
            output_df.loc['average', col] = df[col].astype(float).mean()
            output_df.loc['geometric_mean', col] = np.exp(np.mean(np.log(df[col].astype(float)))) if (df[col].astype(float) > 0).all() else 0.0
        else:
            output_df.loc['average', col] = prefix + 'average'
            output_df.loc['geometric_mean', col] = prefix + 'geometric_mean'
    return output_df.reset_index(drop=True)

# Normalize the values in comparison_df based on baseline_df for rows with matching name in benchmark and task_name
def normalize_matching_rows(baseline_df, comparison_df):
    normalized_df = comparison_df.copy()
    for idx, row in comparison_df.iterrows():
        benchmark = row['benchmark']
        task_name = row['task_name']
        baseline_row = baseline_df[(baseline_df['benchmark'] == benchmark) & (baseline_df['task_name'] == task_name)]
        for col in comparison_df.columns:
            if col not in ['benchmark', 'task_name']:
                if (not baseline_row.empty) and (baseline_row.loc[idx, col] not in ['N/A', -1.0, ""]):
                    try:
                        normalized_value = float(row[col]) / float(baseline_row.loc[idx, col])
                        normalized_df.at[idx, col] = normalized_value
                    except:
                        normalized_df.at[idx, col] = np.nan
                elif (not baseline_row.empty) and (baseline_row.loc[idx, col] == "" and row[col] == ""):
                    normalized_df.at[idx, col] = ""
                else:
                    normalized_df.at[idx, col] = 'N/A'
    return normalized_df

def main(args):
    # Map files to labels
    if len(args.files) != len(args.labels):
        args.labels = [file.split('.')[0] for file in args.files]
        print("Warning: The number of files and labels given do not match. Ignoring labels. Will use file names as labels.")
    file_label_map = dict(zip(args.files, args.labels))

    # Define benchmark groups
    benchmark_groups = []
    benchmark_groups.append(dict(
        benchmark_group_name = "Synthetic-Mesh",
        benchmark_group_patterns = [("microbenchmark", "mesh")]
    ))
    benchmark_groups.append(dict(
        benchmark_group_name = "Synthetic-Line",
        benchmark_group_patterns = [("microbenchmark", "line")]
    ))
    benchmark_groups.append(dict(
        benchmark_group_name = "Synthetic-Tree",
        benchmark_group_patterns = [("microbenchmark", "tree")]
    ))
    benchmark_groups.append(dict(
        benchmark_group_name = "Real_World_Application-Edge_Detection",
        benchmark_group_patterns = [("edge_detection", "")]
    ))
    benchmark_groups.append(dict(
        benchmark_group_name = "Real_World_Application-GEMM",
        benchmark_group_patterns = [("GEMM", ""), ("vector_scalar_mul", "")]
    ))
    benchmark_groups.append(dict(
        benchmark_group_name = "Real_World_Application-ML",
        benchmark_group_patterns = [("ResNet", "")]
    ))
    def helper_get_partial_df(df, benchmark_group_patterns):
        mask = pd.Series([False] * df.shape[0])
        for benchmark_name, task_name_pattern in benchmark_group_patterns:
            mask |= df["benchmark"].str.contains(benchmark_name, na=False) & df["task_name"].str.contains(task_name_pattern, na=False)
        return df[mask].copy()

    # Read CSV files into DataFrames
    baseline_file = args.files[0]
    dfs = {}
    for file in args.files:
        try:
            data = pd.read_csv(file)
            print(f"Successfully read {file} with shape {data.shape}")
            
            # Strip extra space around the value and sort data by benchmark and task_name
            data.columns = data.columns.str.strip()
            data.sort_values(by=['benchmark', 'task_name'], inplace=True, ignore_index=True)
            dfs[file] = data

            # Remove failed test cases which contain 'N/A' in any column except 'benchmark' and 'task_name' or -1.0 for avg_runtime [us]
            mask = (data.drop(columns=['benchmark', 'task_name']).map(lambda x: x == 'N/A').any(axis=1)) | (data['avg_runtime [us]'] == -1.0)
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

    # Calculate averages and geometric means for each common success file by overall and benchmark groups
    for file in args.files:
        df_with_stats = pd.DataFrame(columns=dfs["common_success_" + file].columns)
        empty_row_df = pd.DataFrame([[""]*len(dfs["common_success_" + file].columns)], columns=dfs["common_success_" + file].columns)
        for group in benchmark_groups:
            partial_df = helper_get_partial_df(dfs["common_success_" + file], group['benchmark_group_patterns'])
            if not partial_df.empty:
                partial_df_with_stats = calculate_averages_and_geometric_means(partial_df, prefix=group['benchmark_group_name'] + " ")
                df_with_stats = pd.concat([df_with_stats, partial_df_with_stats, empty_row_df], ignore_index=True)
        dfs["common_success_" + file] = calculate_averages_and_geometric_means(dfs["common_success_" + file], prefix="Overall ")
        dfs["common_success_" + file] = pd.concat([df_with_stats, dfs["common_success_" + file][dfs["common_success_" + file]["benchmark"].str.contains("average|geometric_mean")]], ignore_index=True)

    # Calculate the normalized values
    for file in args.files:
        dfs["normalized_" + file] = normalize_matching_rows(dfs["common_success_" + baseline_file], dfs["common_success_" + file])
    
    # Collect overall stats for each file and calculate stats for each group
    dfs["Overall Statistics"] = pd.DataFrame(columns=['File', 'Benchmark Group', 'Total Number Test Cases', 'Successful Test Cases', 'Success Rate', 'Common Successful Test Cases Across All Files'])
    for file in args.files:
        total_test_cases = dfs[file].shape[0]
        successful_test_cases = dfs["success_" + file].shape[0]
        common_successful_test_cases = dfs["common_success_" + file][~dfs["common_success_" + file]["benchmark"].str.contains("average|geometric_mean")].shape[0]
        for group in benchmark_groups:
            group_total = helper_get_partial_df(dfs[file], group['benchmark_group_patterns']).shape[0]
            group_successful = helper_get_partial_df(dfs["success_" + file], group['benchmark_group_patterns']).shape[0]
            group_common_successful = helper_get_partial_df(dfs["common_success_" + file], group['benchmark_group_patterns']).shape[0]
            dfs["Overall Statistics"] = pd.concat([dfs["Overall Statistics"], pd.Series({
                'File': file,
                'Benchmark Group': group['benchmark_group_name'],
                'Total Number Test Cases': group_total,
                'Successful Test Cases': group_successful,
                'Success Rate': float(group_successful) / float(group_total) if float(group_total) > 0.0 else 0.0,
                'Common Successful Test Cases Across All Files': group_common_successful
            }).to_frame().T], ignore_index=True)
        # Add overall stats
        dfs["Overall Statistics"] = pd.concat([dfs["Overall Statistics"], pd.Series({
            'File': file,
            'Benchmark Group': 'Overall',
            'Total Number Test Cases': total_test_cases,
            'Successful Test Cases': successful_test_cases,
            'Success Rate': float(successful_test_cases) / float(total_test_cases) if float(total_test_cases) > 0.0 else 0.0,
            'Common Successful Test Cases Across All Files': common_successful_test_cases
        }).to_frame().T], ignore_index=True)
    
    with open('output.log', 'w') as output_log:
        output_log.write(str(dfs))

    # Save the results to Excel files
    with pd.ExcelWriter('comparison_results.xlsx', engine='openpyxl') as writer:
        dfs["Overall Statistics"].to_excel(writer, sheet_name="Overall Statistics", index=False)
        for file in args.files:
            baseline_file_name = file_label_map[baseline_file]
            file_name = file_label_map[file]
            dfs[file].to_excel(writer, sheet_name=f"{file_name}"[:31], index=False)
            dfs["success_" + file].to_excel(writer, sheet_name=f"success_{file_name}"[:31], index=False)
            dfs["common_success_" + file].to_excel(writer, sheet_name=f"common_success_{file_name}"[:31], index=False)
            dfs["normalized_" + file].to_excel(writer, sheet_name=f"normalized_{file_name}_vs_{baseline_file_name}"[:31], index=False)

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
        '-l',
        '--labels',
        nargs='+',
        required=False,
        help='List of labels for the CSV files',
        default=[]
    )
    args = parser.parse_args()
    main(args)
