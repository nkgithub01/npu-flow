import argparse
import csv
import numpy as np
import pandas as pd

# Calculate average and geometric mean for each column in a DataFrame and add these values back to the DataFrame
def calculate_averages_and_geometric_means(df):
    output_df = df.copy()
    for col in output_df.columns:
        if col not in ['benchmark', 'task_name']:
            output_df.loc['average', col] = df[col].astype(float).mean()
            output_df.loc['geometric_mean', col] = np.exp(np.mean(np.log(df[col].astype(float)))) if (df[col].astype(float) > 0).all() else 0.0
        else:
            output_df.loc['average', col] = 'average'
            output_df.loc['geometric_mean', col] = 'geometric_mean'
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
                if not baseline_row.empty:
                    try:
                        normalized_value = float(row[col]) / float(baseline_row.iloc[0][col])
                        normalized_df.at[idx, col] = normalized_value
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

    # Calculate averages and geometric means for each file
    for file in args.files:
        dfs["common_success_" + file] = calculate_averages_and_geometric_means(dfs["common_success_" + file])

    # Calculate the normalized values
    for file in args.files:
        dfs["normalized_" + file] = normalize_matching_rows(dfs["common_success_" + baseline_file], dfs["common_success_" + file])
    
    # Collect stats for each file
    dfs["Overall Statistics"] = pd.DataFrame(columns=['File', 'Benchmark Group', 'Total Number Test Cases', 'Successful Test Cases', 'Success Rate', 'Common Successful Test Cases Across All Files'])
    benchmark_groups = []
    benchmark_groups.append(dict(
        benchmark_group_name = "Synthetic-Mesh",
        benchmark_names = ["microbenchmark"],
        task_name_patterns = ["mesh"]
    ))
    benchmark_groups.append(dict(
        benchmark_group_name = "Synthetic-Line",
        benchmark_names = ["microbenchmark"],
        task_name_patterns = ["line"]
    ))
    benchmark_groups.append(dict(
        benchmark_group_name = "Synthetic-Tree",
        benchmark_names = ["microbenchmark"],
        task_name_patterns = ["tree"]
    ))
    benchmark_groups.append(dict(
        benchmark_group_name = "Real_World_Application-Edge_Detection",
        benchmark_names = ["edge_detection"],
        task_name_patterns = [""]
    ))
    benchmark_groups.append(dict(
        benchmark_group_name = "Real_World_Application-GEMM",
        benchmark_names = ["GEMM", "vector_scalar_mul"],
        task_name_patterns = ["", ""]
    ))
    benchmark_groups.append(dict(
        benchmark_group_name = "Real_World_Application-ML",
        benchmark_names = ["ResNet"],
        task_name_patterns = [""]
    ))
    def helper_get_partial_df(df, benchmark_names, task_name_patterns):
        if len(benchmark_names) != len(task_name_patterns):
            raise ValueError("Length of benchmark_names and task_name_patterns must be the same")
        mask = pd.Series([False] * df.shape[0])
        for benchmark_name, task_name_pattern in zip(benchmark_names, task_name_patterns):
            mask |= df["benchmark"].str.contains(benchmark_name, na=False) & df["task_name"].str.contains(task_name_pattern, na=False)
        return df[mask].copy()

    # Calculate stats for each group
    for file in args.files:
        total_test_cases = dfs[file].shape[0]
        successful_test_cases = dfs["success_" + file].shape[0]
        common_successful_test_cases = dfs["common_success_" + file][~dfs["common_success_" + file]["benchmark"].str.contains("average|geometric_mean")].shape[0]
        for group in benchmark_groups:
            group_total = helper_get_partial_df(dfs[file], group['benchmark_names'], group['task_name_patterns']).shape[0]
            group_successful = helper_get_partial_df(dfs["success_" + file], group['benchmark_names'], group['task_name_patterns']).shape[0]
            group_common_successful = helper_get_partial_df(dfs["common_success_" + file], group['benchmark_names'], group['task_name_patterns']).shape[0]
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

    # Save the results to Excel files
    with pd.ExcelWriter('comparison_results.xlsx', engine='openpyxl') as writer:
        dfs["Overall Statistics"].to_excel(writer, sheet_name="Overall Statistics", index=False)
        for file in args.files:
            baseline_file_name = baseline_file.split('.')[0]
            file_name = file.split('.')[0]
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
    args = parser.parse_args()
    main(args)
