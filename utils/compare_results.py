import os
import re
import argparse
import csv
import math
import numpy as np
import pandas as pd
from openpyxl.styles import PatternFill, Border, Side

# Calculate average and geometric mean for each column in a DataFrame and add these values back to the DataFrame
def calculate_averages_and_geometric_means(df, prefix=""):
    if len(df) == 0:
        return df
    output_df = df.copy()
    for col in output_df.columns:
        if col not in ["benchmark_name", 'task_name']:
            output_df.loc['average', col] = df[col].astype(float).mean()
            output_df.loc['geometric_mean', col] = np.exp(np.mean(np.log(df[col].astype(float)))) if (df[col].astype(float) > 0).all() else np.nan
        else:
            output_df.loc['average', col] = prefix + 'average'
            output_df.loc['geometric_mean', col] = prefix + 'geometric_mean'
    return output_df.reset_index(drop=True)

# Normalize the values in comparison_df based on baseline_df for rows with matching name in benchmark and task_name
def normalize_matching_rows(baseline_df, comparison_df):
    normalized_df = comparison_df.copy()
    for idx, row in comparison_df.iterrows():
        benchmark = row["benchmark_name"]
        task_name = row['task_name']
        baseline_row = baseline_df[(baseline_df["benchmark_name"] == benchmark) & (baseline_df['task_name'] == task_name)]
        for col in comparison_df.columns:
            if col not in ["benchmark_name", 'task_name']:
                if (not baseline_row.empty) and (baseline_row.loc[idx, col] not in ['N/A', -1.0, "", float('nan')]):
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
    # Check if the number of files and labels match
    if len(args.files) != len(args.labels):
        args.labels = [file.split('.')[0] for file in args.files]
        print("Warning: The number of files and labels given do not match. Ignoring labels. Will use file names as labels.")
    
    # Check if all files exist, if not, remove them from the list
    tmp_file_list = []
    tmp_label_list = []
    for idx, file in enumerate(args.files):
        if not os.path.isfile(file):
            print(f"Error: File {file} does not exist.")
        else:
            tmp_file_list.append(file)
            tmp_label_list.append(args.labels[idx])
    if len(tmp_file_list) == 0:
        print("Error: No valid files to process. Exiting.")
        return
    
    # Map files to labels
    args.files = tmp_file_list
    args.labels = tmp_label_list
    file_label_map = dict(zip(args.files, args.labels))

    # Define benchmark groups
    benchmark_groups = []
    benchmark_groups.append(dict(
        benchmark_group_name = "Synthetic-Mesh",
        benchmark_group_patterns = [("microbenchmark", "mesh"), ("microbenchmark_with_feedback_loop", "mesh")]
    ))
    benchmark_groups.append(dict(
        benchmark_group_name = "Synthetic-Line",
        benchmark_group_patterns = [("microbenchmark", "line"), ("microbenchmark_with_feedback_loop", "line")]
    ))
    benchmark_groups.append(dict(
        benchmark_group_name = "Synthetic-Tree",
        benchmark_group_patterns = [("microbenchmark", "tree"), ("microbenchmark_with_feedback_loop", "tree")]
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
    def helper_get_benchmark_group_df(df, benchmark_group_patterns):
        mask = pd.Series([False] * df.shape[0])
        for benchmark_name, task_name_pattern in benchmark_group_patterns:
            mask |= df["benchmark_name"].str.contains(benchmark_name, na=False) & df["task_name"].str.contains(task_name_pattern, na=False)
        return df[mask].copy()
    def helper_get_ungrouped_df(df):
        df.reset_index(drop=True, inplace=True)
        mask = pd.Series([True] * df.shape[0])
        for group in benchmark_groups:
            for benchmark_name, task_name_pattern in group['benchmark_group_patterns']:
                mask &= ~(df["benchmark_name"].str.contains(benchmark_name, na=False) & df["task_name"].str.contains(task_name_pattern, na=False))
        return df[mask].copy()

    # Read CSV files into DataFrames
    baseline_file = args.files[0]
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

            # Remove failed test cases which contain 'N/A' in any column except "benchmark_name" and 'task_name' or -1.0 for avg_runtime [us]
            mask = (data['compilation_time [s]'].map(lambda x: math.isnan(x)))
            dfs["success_" + file] = data[~mask].reset_index(drop=True)
            print(f"After removing failed test cases, {file} has shape {dfs["success_" + file].shape}")

        except Exception as e:
            print(f"Error reading {file}: {e}")

    # Create dataframes that contain only the test cases that are present in all success files
    common_benchmarks = set(dfs["success_" + baseline_file][["benchmark_name", 'task_name']].itertuples(index=False, name=None))
    for file in args.files:
        if file != baseline_file:
            common_benchmarks.intersection_update(set(dfs["success_" + file][["benchmark_name", 'task_name']].itertuples(index=False, name=None)))
    for file in args.files:
        dfs["common_success_" + file] = dfs["success_" + file][dfs["success_" + file][["benchmark_name", 'task_name']].apply(tuple, axis=1).isin(common_benchmarks)].reset_index(drop=True)
        print(f"After filtering to common benchmarks that all files have success run results, {file} has shape {dfs["common_success_" + file].shape}")

    # Calculate averages and geometric means for each common success file by overall and benchmark groups
    for file in args.files:
        df_with_stats = pd.DataFrame(columns=dfs["common_success_" + file].columns)
        empty_row_df = pd.DataFrame([[""]*len(dfs["common_success_" + file].columns)], columns=dfs["common_success_" + file].columns)
        for group in benchmark_groups:
            partial_df = helper_get_benchmark_group_df(dfs["common_success_" + file], group['benchmark_group_patterns'])
            if not partial_df.empty:
                partial_df_with_stats = calculate_averages_and_geometric_means(partial_df, prefix=group['benchmark_group_name'] + " ")
                df_with_stats = pd.concat([df_with_stats, partial_df_with_stats, empty_row_df], ignore_index=True)
        ungrouped_df = helper_get_ungrouped_df(dfs["common_success_" + file])
        if not ungrouped_df.empty:
            ungrouped_df_with_stats = calculate_averages_and_geometric_means(ungrouped_df, prefix="Ungrouped ")
            df_with_stats = pd.concat([df_with_stats, ungrouped_df_with_stats, empty_row_df], ignore_index=True)
        # Add overall stats
        dfs["common_success_" + file] = calculate_averages_and_geometric_means(dfs["common_success_" + file], prefix="Overall ")
        dfs["common_success_" + file] = pd.concat([df_with_stats, empty_row_df, dfs["common_success_" + file][dfs["common_success_" + file]["benchmark_name"].str.contains("average|geometric_mean")]], ignore_index=True)

    # Calculate the normalized values
    for file in args.files:
        dfs["normalized_" + file] = normalize_matching_rows(dfs["common_success_" + baseline_file], dfs["common_success_" + file])
    
    # Collect overall stats for each file and calculate stats for each group
    dfs["Overall Statistics"] = pd.DataFrame(columns=['Sheet', 'Benchmark Group', 'Total Number Test Cases', 'Number of Successfully Builded Test Cases', 'Number of Successfully Compiled Test Cases', 'Successful Test Cases', 'Success Rate', 'Common Successful Test Cases Across All Files'])
    empty_row_df = pd.DataFrame([[""]*len(dfs["Overall Statistics"].columns)], columns=dfs["Overall Statistics"].columns)
    for file in args.files:
        total_test_cases = dfs[file].shape[0]
        successful_builded = dfs[file][dfs[file]['build_time [s]'].map(lambda x: not math.isnan(x))].shape[0]
        successful_compiled = dfs[file][dfs[file]['compilation_time [s]'].map(lambda x: not math.isnan(x))].shape[0]
        successful_test_cases = dfs["success_" + file].shape[0]
        common_successful_test_cases = dfs["common_success_" + file][~(dfs["common_success_" + file]["benchmark_name"].str.contains("average|geometric_mean") | dfs["common_success_" + file]["benchmark_name"].str.match(r"^$"))].shape[0]
        for group in benchmark_groups:
            gropu_df = helper_get_benchmark_group_df(dfs[file], group['benchmark_group_patterns'])
            group_total = gropu_df.shape[0]
            group_successful_builded = gropu_df[gropu_df['build_time [s]'].map(lambda x: not math.isnan(x))].shape[0]
            group_successful_compiled = gropu_df[gropu_df['compilation_time [s]'].map(lambda x: not math.isnan(x))].shape[0]
            group_successful = helper_get_benchmark_group_df(dfs["success_" + file], group['benchmark_group_patterns']).shape[0]
            group_common_successful = helper_get_benchmark_group_df(dfs["common_success_" + file], group['benchmark_group_patterns']).shape[0]
            dfs["Overall Statistics"] = pd.concat([dfs["Overall Statistics"], pd.Series({
                'Sheet': file_label_map[file],
                'Benchmark Group': group['benchmark_group_name'],
                'Total Number Test Cases': group_total,
                'Number of Successfully Builded Test Cases': group_successful_builded,
                'Number of Successfully Compiled Test Cases': group_successful_compiled,
                'Successful Test Cases': group_successful,
                'Success Rate': float(group_successful) / float(group_total) if float(group_total) > 0.0 else 0.0,
                'Common Successful Test Cases Across All Files': group_common_successful
                }).to_frame().T
            ], ignore_index=True)
        # Add ungrouped stats
        ungrouped_df = helper_get_ungrouped_df(dfs[file])
        ungrouped_total = ungrouped_df.shape[0]
        ungroup_successful_builded = ungrouped_df[ungrouped_df['build_time [s]'].map(lambda x: not math.isnan(x))].shape[0]
        ungroup_successful_compiled = ungrouped_df[ungrouped_df['compilation_time [s]'].map(lambda x: not math.isnan(x))].shape[0]
        ungrouped_successful = helper_get_ungrouped_df(dfs["success_" + file]).shape[0]
        ungrouped_common_successful = helper_get_ungrouped_df(dfs["common_success_" + file][~(dfs["common_success_" + file]["benchmark_name"].str.contains("average|geometric_mean") | dfs["common_success_" + file]["benchmark_name"].str.match(r"^$"))]).shape[0]
        dfs["Overall Statistics"] = pd.concat([dfs["Overall Statistics"], pd.Series({
            'Sheet': file_label_map[file],
            'Benchmark Group': 'Ungrouped',
            'Total Number Test Cases': ungrouped_total,
            'Number of Successfully Builded Test Cases': ungroup_successful_builded,
            'Number of Successfully Compiled Test Cases': ungroup_successful_compiled,
            'Successful Test Cases': ungrouped_successful,
            'Success Rate': float(ungrouped_successful) / float(ungrouped_total) if float(ungrouped_total) > 0.0 else 0.0,
            'Common Successful Test Cases Across All Files': ungrouped_common_successful
            }).to_frame().T
        ], ignore_index=True)    
        # Add overall stats
        dfs["Overall Statistics"] = pd.concat([dfs["Overall Statistics"], pd.Series({
            'Sheet': file_label_map[file],
            'Benchmark Group': 'Overall',
            'Total Number Test Cases': total_test_cases,
            'Number of Successfully Builded Test Cases': successful_builded,
            'Number of Successfully Compiled Test Cases': successful_compiled,
            'Successful Test Cases': successful_test_cases,
            'Success Rate': float(successful_test_cases) / float(total_test_cases) if float(total_test_cases) > 0.0 else 0.0,
            'Common Successful Test Cases Across All Files': common_successful_test_cases
            }).to_frame().T,
            empty_row_df
        ], ignore_index=True)
    
    with open('output.log', 'w') as output_log:
        output_log.write(str(dfs))

    # Save the results to Excel files
    with pd.ExcelWriter(args.output_file, engine='openpyxl') as writer:
        dfs["Overall Statistics"].to_excel(writer, sheet_name="Overall Statistics", index=False)
        for file in args.files:
            baseline_file_name = file_label_map[baseline_file]
            file_name = file_label_map[file]
            dfs[file].to_excel(writer, sheet_name=f"{file_name}"[:31], index=False)
            dfs["success_" + file].to_excel(writer, sheet_name=f"success_{file_name}"[:31], index=False)
            dfs["common_success_" + file].to_excel(writer, sheet_name=f"common_success_{file_name}"[:31], index=False)
            dfs["normalized_" + file].to_excel(writer, sheet_name=f"normalized_{file_name}_vs_{baseline_file_name}"[:31], index=False)

        # Color the 'Overall Statistics' sheet, color the Overall row with light green
        worksheet = writer.sheets["Overall Statistics"]
        light_green_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
        for row in range(2, worksheet.max_row + 1):
            if worksheet.cell(row=row, column=2).value == "Overall":
                for col in range(1, worksheet.max_column + 1):
                    worksheet.cell(row=row, column=col).fill = light_green_fill

        # Color the average and geometric_mean rows in each sheet with light green
        for file in args.files:
            file_name = file_label_map[file]
            worksheet = writer.sheets[f"common_success_{file_name}"[:31]]
            for row in range(2, worksheet.max_row + 1):
                if any(item in worksheet.cell(row=row, column=2).value for item in ["average", "geometric_mean"]):
                    for col in range(1, worksheet.max_column + 1):
                        worksheet.cell(row=row, column=col).fill = light_green_fill

        # Color the cell base on the value in normalized sheets, green to red gradient from 0 to 10
        for file in args.files:
            file_name = file_label_map[file]
            baseline_file_name = file_label_map[baseline_file]
            worksheet = writer.sheets[f"normalized_{file_name}_vs_{baseline_file_name}"[:31]]
            for row in range(2, worksheet.max_row + 1):
                for col in range(3, worksheet.max_column + 1):
                    cell_value = worksheet.cell(row=row, column=col).value
                    if isinstance(cell_value, (int, float)) and float(cell_value) != 1.0:
                        # Map the value to a color
                        R0, G0, B0 = 0x00, 0xAF, 0x00  # green
                        R1, G1, B1 = 0xFF, 0xFF, 0xFF  # white
                        R2, G2, B2 = 0xAF, 0x00, 0x00  # red
                        if cell_value <= 1.0:
                            # Gradient from green (0.0) to white (1.0)
                            # 0.0 -> green (00AF00), 1.0 -> white (FFFFFF)
                            scale = cell_value
                            red_intensity = int(R0 + (R1 - R0) * scale)
                            green_intensity = int(G0 + (G1 - G0) * scale)
                            blue_intensity = int(B0 + (B1 - B0) * scale)
                        elif cell_value <= 10.0:
                            # Gradient from white (1.0) to red (10.0)
                            # 1.0 -> white (FFFFFF), 10.0 -> red (FF0000)
                            scale = cell_value / 10.0
                            red_intensity = int(R1 + (R2 - R1) * scale)
                            green_intensity = int(G1 + (G2 - G1) * scale)
                            blue_intensity = int(B1 + (B2 - B1) * scale)
                        else:
                            # Bright red for values > 2.0
                            red_intensity = R2
                            green_intensity = G2
                            blue_intensity = B2
                        fill_color = f"{red_intensity:02X}{green_intensity:02X}{blue_intensity:02X}"
                        fill = PatternFill(start_color=fill_color, end_color=fill_color, fill_type="solid")
                        worksheet.cell(row=row, column=col).fill = fill

        # Add border to the average and geometric_mean rows in each sheet with orange border
        for file in args.files:
            file_name = file_label_map[file]
            baseline_file_name = file_label_map[baseline_file]
            worksheet = writer.sheets[f"common_success_{file_name}"[:31]]
            for row in range(2, worksheet.max_row + 1):
                if any(item in worksheet.cell(row=row, column=2).value for item in ["average", "geometric_mean"]):
                    for col in range(1, worksheet.max_column + 1):
                        thick_border_side = Side(border_style="thick", color="FFA500") # orange border
                        worksheet.cell(row=row, column=col).border = Border(
                            left=thick_border_side,
                            right=thick_border_side,
                            top=thick_border_side,
                            bottom=thick_border_side
                        )
            worksheet = writer.sheets[f"normalized_{file_name}_vs_{baseline_file_name}"[:31]]
            for row in range(2, worksheet.max_row + 1):
                if any(item in worksheet.cell(row=row, column=2).value for item in ["average", "geometric_mean"]):
                    for col in range(1, worksheet.max_column + 1):
                        thick_border_side = Side(border_style="thick", color="FFA500") # orange border
                        worksheet.cell(row=row, column=col).border = Border(
                            left=thick_border_side,
                            right=thick_border_side,
                            top=thick_border_side,
                            bottom=thick_border_side
                        )
        
        # Freeze the first row and the first 2 columns of the sheet
        for file in args.files:
            file_name = file_label_map[file]
            baseline_file_name = file_label_map[baseline_file]
            writer.sheets[f"{file_name}"[:31]].freeze_panes = "C2"
            writer.sheets[f"success_{file_name}"[:31]].freeze_panes = "C2"
            writer.sheets[f"common_success_{file_name}"[:31]].freeze_panes = "C2"
            writer.sheets[f"normalized_{file_name}_vs_{baseline_file_name}"[:31]].freeze_panes = "C2"
        writer.sheets["Overall Statistics"].freeze_panes = "C2"
    
    print(f"Comparison results saved to {args.output_file}")

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
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        dest='output_file',
        required=False,
        help='Output file for the comparison results',
        default='comparison_results.xlsx'
    )
    args = parser.parse_args()
    main(args)
