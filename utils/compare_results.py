import os
import re
import argparse
import csv
import math
import numpy as np
import pandas as pd
from openpyxl.styles import PatternFill, Border, Side

benchmark_groups_by_type = []
benchmark_groups_by_type.append(dict(
    benchmark_group_name = "Synthetic-Line",
    benchmark_group_patterns = [("microbenchmark", "line"), ("microbenchmark_with_feedback_loop", "line")]
))
benchmark_groups_by_type.append(dict(
    benchmark_group_name = "Synthetic-Mesh",
    benchmark_group_patterns = [("microbenchmark", "mesh"), ("microbenchmark_with_feedback_loop", "mesh"), ("microbenchmark", "Custom_CNN")]
))
benchmark_groups_by_type.append(dict(
    benchmark_group_name = "Synthetic-Tree",
    benchmark_group_patterns = [("microbenchmark", "tree"), ("microbenchmark_with_feedback_loop", "tree"), ("microbenchmark", "single_multicast")]
))
benchmark_groups_by_type.append(dict(
    benchmark_group_name = "Real_World_Application-Edge_Detection",
    benchmark_group_patterns = [("edge_detection", "")]
))
benchmark_groups_by_type.append(dict(
    benchmark_group_name = "Real_World_Application-GEMM",
    benchmark_group_patterns = [("GEMM", ""), ("vector_scalar_mul", "")]
))
benchmark_groups_by_type.append(dict(
    benchmark_group_name = "Real_World_Application-ML",
    benchmark_group_patterns = [("ResNet", "")]
))

benchmark_groups_synth_vs_real_world = []
benchmark_groups_synth_vs_real_world.append(dict(
    benchmark_group_name = "Synthetic",
    benchmark_group_patterns = [
        ("microbenchmark", "line"),
        ("microbenchmark_with_feedback_loop", "line"),
        ("microbenchmark", "mesh"),
        ("microbenchmark_with_feedback_loop", "mesh"),
        ("microbenchmark", "Custom_CNN"),
        ("microbenchmark", "tree"),
        ("microbenchmark_with_feedback_loop", "tree"),
        ("microbenchmark", "single_multicast")
    ]
))
benchmark_groups_synth_vs_real_world.append(dict(
    benchmark_group_name = "Real_World_Application",
    benchmark_group_patterns = [
        ("edge_detection", ""),
        ("GEMM", ""),
        ("vector_scalar_mul", ""),
        ("ResNet", "")
    ]
))

benchmark_groups_pipelined_vs_feedback_loop = []
benchmark_groups_pipelined_vs_feedback_loop.append(dict(
    benchmark_group_name = "Pipelined",
    benchmark_group_patterns = [
        ("microbenchmark", ""),
        ("edge_detection", ""),
        ("GEMM", ""),
        ("vector_scalar_mul", ""),
        ("ResNet", "")
    ]
))
benchmark_groups_pipelined_vs_feedback_loop.append(dict(
    benchmark_group_name = "Feedback_Loop",
    benchmark_group_patterns = [
        ("microbenchmark_with_feedback_loop", "")
    ]
))

benchmark_groups_by_size = []
benchmark_groups_by_size.append(dict(
    benchmark_group_name = "Small-Test-Cases",
    benchmark_group_patterns = [
        ("microbenchmark", "R3_C1"),
        ("microbenchmark", "R4_C1"),
        ("microbenchmark", "R5_C1"),
        ("microbenchmark", "R6_C1"),
        ("microbenchmark", "R3_C2"),
        ("microbenchmark", "R4_C2"),
        ("microbenchmark", "R5_C2"),
        ("microbenchmark", "R6_C2"),
        ("microbenchmark", "R3_C3"),
        ("microbenchmark", "R4_C3"),
        ("microbenchmark", "R5_C3"),
        ("microbenchmark", "R6_C3"),
        ("microbenchmark", "R3_C4"),
        ("microbenchmark", "R4_C4"),
        ("microbenchmark", "R5_C4"),
        ("microbenchmark", "R6_C4"),
        ("microbenchmark", "single_multicast"),
        ("microbenchmark_with_feedback_loop", "R3_C1"),
        ("microbenchmark_with_feedback_loop", "R4_C1"),
        ("microbenchmark_with_feedback_loop", "R5_C1"),
        ("microbenchmark_with_feedback_loop", "R6_C1"),
        ("microbenchmark_with_feedback_loop", "R3_C2"),
        ("microbenchmark_with_feedback_loop", "R4_C2"),
        ("microbenchmark_with_feedback_loop", "R5_C2"),
        ("microbenchmark_with_feedback_loop", "R6_C2"),
        ("microbenchmark_with_feedback_loop", "R3_C3"),
        ("microbenchmark_with_feedback_loop", "R4_C3"),
        ("microbenchmark_with_feedback_loop", "R5_C3"),
        ("microbenchmark_with_feedback_loop", "R6_C3"),
        ("microbenchmark_with_feedback_loop", "R3_C4"),
        ("microbenchmark_with_feedback_loop", "R4_C4"),
        ("microbenchmark_with_feedback_loop", "R5_C4"),
        ("microbenchmark_with_feedback_loop", "R6_C4"),
        ("microbenchmark_with_feedback_loop", "line_length_4"),
        ("microbenchmark_with_feedback_loop", "line_length_5"),
        ("microbenchmark_with_feedback_loop", "line_length_6"),
        ("microbenchmark_with_feedback_loop", "line_length_7"),
        ("microbenchmark_with_feedback_loop", "line_length_8"),
        ("microbenchmark_with_feedback_loop", "line_length_9"),
        ("microbenchmark_with_feedback_loop", "line_length_10"),
        ("microbenchmark_with_feedback_loop", "line_length_11"),
        ("microbenchmark_with_feedback_loop", "line_length_12"),
        ("microbenchmark_with_feedback_loop", "line_length_13"),
        ("microbenchmark_with_feedback_loop", "line_length_14"),
        ("microbenchmark_with_feedback_loop", "line_length_15"),
        ("microbenchmark_with_feedback_loop", "line_length_16"),
        ("microbenchmark_with_feedback_loop", "line_length_17"),
        ("microbenchmark_with_feedback_loop", "line_length_18"),
        ("edge_detection", "col_1"),
        ("edge_detection", "col_2"),
        ("edge_detection", "col_3"),
        ("edge_detection", "col_4"),
        ("GEMM", "R6_C1"),
        ("GEMM", "R6_C2"),
        ("vector_scalar_mul", "")
    ]
))
benchmark_groups_by_size.append(dict(
    benchmark_group_name = "Large-Test-Cases",
    benchmark_group_patterns = [
        ("microbenchmark", "R3_C5"),
        ("microbenchmark", "R4_C5"),
        ("microbenchmark", "R5_C5"),
        ("microbenchmark", "R6_C5"),
        ("microbenchmark", "R3_C6"),
        ("microbenchmark", "R4_C6"),
        ("microbenchmark", "R5_C6"),
        ("microbenchmark", "R6_C6"),
        ("microbenchmark", "R3_C7"),
        ("microbenchmark", "R4_C7"),
        ("microbenchmark", "R5_C7"),
        ("microbenchmark", "R6_C7"),
        ("microbenchmark", "R3_C8"),
        ("microbenchmark", "R4_C8"),
        ("microbenchmark", "R5_C8"),
        ("microbenchmark", "R6_C8"),
        ("microbenchmark", "3d_mesh"),
        ("microbenchmark", "Custom_CNN"),
        ("microbenchmark_with_feedback_loop", "R3_C5"),
        ("microbenchmark_with_feedback_loop", "R4_C5"),
        ("microbenchmark_with_feedback_loop", "R5_C5"),
        ("microbenchmark_with_feedback_loop", "R6_C5"),
        ("microbenchmark_with_feedback_loop", "R3_C6"),
        ("microbenchmark_with_feedback_loop", "R4_C6"),
        ("microbenchmark_with_feedback_loop", "R5_C6"),
        ("microbenchmark_with_feedback_loop", "R6_C6"),
        ("microbenchmark_with_feedback_loop", "R3_C7"),
        ("microbenchmark_with_feedback_loop", "R4_C7"),
        ("microbenchmark_with_feedback_loop", "R5_C7"),
        ("microbenchmark_with_feedback_loop", "R6_C7"),
        ("microbenchmark_with_feedback_loop", "R3_C8"),
        ("microbenchmark_with_feedback_loop", "R4_C8"),
        ("microbenchmark_with_feedback_loop", "R5_C8"),
        ("microbenchmark_with_feedback_loop", "R6_C8"),
        ("microbenchmark_with_feedback_loop", "line_length_19"),
        ("microbenchmark_with_feedback_loop", "line_length_20"),
        ("microbenchmark_with_feedback_loop", "line_length_21"),
        ("microbenchmark_with_feedback_loop", "line_length_22"),
        ("microbenchmark_with_feedback_loop", "line_length_23"),
        ("microbenchmark_with_feedback_loop", "line_length_24"),
        ("microbenchmark_with_feedback_loop", "line_length_25"),
        ("microbenchmark_with_feedback_loop", "line_length_26"),
        ("microbenchmark_with_feedback_loop", "line_length_27"),
        ("microbenchmark_with_feedback_loop", "line_length_28"),
        ("microbenchmark_with_feedback_loop", "line_length_29"),
        ("microbenchmark_with_feedback_loop", "line_length_30"),
        ("microbenchmark_with_feedback_loop", "line_length_31"),
        ("microbenchmark_with_feedback_loop", "line_length_32"),
        ("edge_detection", "col_5"),
        ("edge_detection", "col_6"),
        ("edge_detection", "col_7"),
        ("edge_detection", "col_8"),
        ("GEMM", "R6_C4"),
        ("GEMM", "R6_C8"),
        ("ResNet", "")
    ]
))

benchmark_groups_pipelined_vs_feedback_loop_by_size = []
benchmark_groups_pipelined_vs_feedback_loop_by_size.append(dict(
    benchmark_group_name = "Small Synthetic Pipeline",
    benchmark_group_patterns = [
        ("microbenchmark", "R3_C1"),
        ("microbenchmark", "R4_C1"),
        ("microbenchmark", "R5_C1"),
        ("microbenchmark", "R6_C1"),
        ("microbenchmark", "R3_C2"),
        ("microbenchmark", "R4_C2"),
        ("microbenchmark", "R5_C2"),
        ("microbenchmark", "R6_C2"),
        ("microbenchmark", "R3_C3"),
        ("microbenchmark", "R4_C3"),
        ("microbenchmark", "R5_C3"),
        ("microbenchmark", "R6_C3"),
        ("microbenchmark", "R3_C4"),
        ("microbenchmark", "R4_C4"),
        ("microbenchmark", "R5_C4"),
        ("microbenchmark", "R6_C4"),
        ("microbenchmark", "single_multicast")
    ]
))
benchmark_groups_pipelined_vs_feedback_loop_by_size.append(dict(
    benchmark_group_name = "Large Synthetic Pipeline",
    benchmark_group_patterns = [
        ("microbenchmark", "R3_C5"),
        ("microbenchmark", "R4_C5"),
        ("microbenchmark", "R5_C5"),
        ("microbenchmark", "R6_C5"),
        ("microbenchmark", "R3_C6"),
        ("microbenchmark", "R4_C6"),
        ("microbenchmark", "R5_C6"),
        ("microbenchmark", "R6_C6"),
        ("microbenchmark", "R3_C7"),
        ("microbenchmark", "R4_C7"),
        ("microbenchmark", "R5_C7"),
        ("microbenchmark", "R6_C7"),
        ("microbenchmark", "R3_C8"),
        ("microbenchmark", "R4_C8"),
        ("microbenchmark", "R5_C8"),
        ("microbenchmark", "R6_C8"),
        ("microbenchmark", "3d_mesh"),
        ("microbenchmark", "Custom_CNN")
    ]
))
benchmark_groups_pipelined_vs_feedback_loop_by_size.append(dict(
    benchmark_group_name = "Small Synthetic feedback loop",
    benchmark_group_patterns = [
        ("microbenchmark_with_feedback_loop", "R3_C1"),
        ("microbenchmark_with_feedback_loop", "R4_C1"),
        ("microbenchmark_with_feedback_loop", "R5_C1"),
        ("microbenchmark_with_feedback_loop", "R6_C1"),
        ("microbenchmark_with_feedback_loop", "R3_C2"),
        ("microbenchmark_with_feedback_loop", "R4_C2"),
        ("microbenchmark_with_feedback_loop", "R5_C2"),
        ("microbenchmark_with_feedback_loop", "R6_C2"),
        ("microbenchmark_with_feedback_loop", "R3_C3"),
        ("microbenchmark_with_feedback_loop", "R4_C3"),
        ("microbenchmark_with_feedback_loop", "R5_C3"),
        ("microbenchmark_with_feedback_loop", "R6_C3"),
        ("microbenchmark_with_feedback_loop", "R3_C4"),
        ("microbenchmark_with_feedback_loop", "R4_C4"),
        ("microbenchmark_with_feedback_loop", "R5_C4"),
        ("microbenchmark_with_feedback_loop", "R6_C4"),
        ("microbenchmark_with_feedback_loop", "line_length_4"),
        ("microbenchmark_with_feedback_loop", "line_length_5"),
        ("microbenchmark_with_feedback_loop", "line_length_6"),
        ("microbenchmark_with_feedback_loop", "line_length_7"),
        ("microbenchmark_with_feedback_loop", "line_length_8"),
        ("microbenchmark_with_feedback_loop", "line_length_9"),
        ("microbenchmark_with_feedback_loop", "line_length_10"),
        ("microbenchmark_with_feedback_loop", "line_length_11"),
        ("microbenchmark_with_feedback_loop", "line_length_12"),
        ("microbenchmark_with_feedback_loop", "line_length_13"),
        ("microbenchmark_with_feedback_loop", "line_length_14"),
        ("microbenchmark_with_feedback_loop", "line_length_15"),
        ("microbenchmark_with_feedback_loop", "line_length_16"),
        ("microbenchmark_with_feedback_loop", "line_length_17"),
        ("microbenchmark_with_feedback_loop", "line_length_18")
    ]
))
benchmark_groups_pipelined_vs_feedback_loop_by_size.append(dict(
    benchmark_group_name = "Large Synthetic feedback loop",
    benchmark_group_patterns = [
        ("microbenchmark_with_feedback_loop", "R3_C5"),
        ("microbenchmark_with_feedback_loop", "R4_C5"),
        ("microbenchmark_with_feedback_loop", "R5_C5"),
        ("microbenchmark_with_feedback_loop", "R6_C5"),
        ("microbenchmark_with_feedback_loop", "R3_C6"),
        ("microbenchmark_with_feedback_loop", "R4_C6"),
        ("microbenchmark_with_feedback_loop", "R5_C6"),
        ("microbenchmark_with_feedback_loop", "R6_C6"),
        ("microbenchmark_with_feedback_loop", "R3_C7"),
        ("microbenchmark_with_feedback_loop", "R4_C7"),
        ("microbenchmark_with_feedback_loop", "R5_C7"),
        ("microbenchmark_with_feedback_loop", "R6_C7"),
        ("microbenchmark_with_feedback_loop", "R3_C8"),
        ("microbenchmark_with_feedback_loop", "R4_C8"),
        ("microbenchmark_with_feedback_loop", "R5_C8"),
        ("microbenchmark_with_feedback_loop", "R6_C8"),
        ("microbenchmark_with_feedback_loop", "line_length_19"),
        ("microbenchmark_with_feedback_loop", "line_length_20"),
        ("microbenchmark_with_feedback_loop", "line_length_21"),
        ("microbenchmark_with_feedback_loop", "line_length_22"),
        ("microbenchmark_with_feedback_loop", "line_length_23"),
        ("microbenchmark_with_feedback_loop", "line_length_24"),
        ("microbenchmark_with_feedback_loop", "line_length_25"),
        ("microbenchmark_with_feedback_loop", "line_length_26"),
        ("microbenchmark_with_feedback_loop", "line_length_27"),
        ("microbenchmark_with_feedback_loop", "line_length_28"),
        ("microbenchmark_with_feedback_loop", "line_length_29"),
        ("microbenchmark_with_feedback_loop", "line_length_30"),
        ("microbenchmark_with_feedback_loop", "line_length_31"),
        ("microbenchmark_with_feedback_loop", "line_length_32")
    ]
))
benchmark_groups_pipelined_vs_feedback_loop_by_size.append(dict(
    benchmark_group_name = "Small Real_World_Application",
    benchmark_group_patterns = [
        ("edge_detection", "col_1"),
        ("edge_detection", "col_2"),
        ("edge_detection", "col_3"),
        ("edge_detection", "col_4"),
        ("GEMM", "R6_C1"),
        ("GEMM", "R6_C2"),
        ("vector_scalar_mul", "")
    ]
))
benchmark_groups_pipelined_vs_feedback_loop_by_size.append(dict(
    benchmark_group_name = "Large Real_World_Application",
    benchmark_group_patterns = [
        ("edge_detection", "col_5"),
        ("edge_detection", "col_6"),
        ("edge_detection", "col_7"),
        ("edge_detection", "col_8"),
        ("GEMM", "R6_C4"),
        ("GEMM", "R6_C8"),
        ("ResNet", "")
    ]
))

def augment_stats(df, prefix=""):
    if len(df) == 0:
        return df
    output_df = df.copy()

    # Find Min and Max for each column in a DataFrame and add these values back to the DataFrame
    for col in output_df.columns:
        if col not in ["benchmark_name", 'task_name']:
            output_df.loc['min', col] = df[col].astype(float).min()
            output_df.loc['max', col] = df[col].astype(float).max()
        else:
            output_df.loc['min', col] = prefix + 'min'
            output_df.loc['max', col] = prefix + 'max'

    # Calculate average and geometric mean for each column in a DataFrame and add these values back to the DataFrame
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
    normalized_df = comparison_df.copy().astype(object)
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
    def helper_get_benchmark_group_df(df, benchmark_group_patterns):
        mask = pd.Series([False] * df.shape[0])
        for benchmark_name, task_name_pattern in benchmark_group_patterns:
            mask |= (df["benchmark_name"] == benchmark_name) & df["task_name"].str.contains(task_name_pattern, na=False)
        return df[mask].copy()
    def helper_get_ungrouped_df(df):
        df.reset_index(drop=True, inplace=True)
        mask = pd.Series([True] * df.shape[0])
        for group in benchmark_groups:
            for benchmark_name, task_name_pattern in group['benchmark_group_patterns']:
                mask &= ~((df["benchmark_name"] == benchmark_name) & df["task_name"].str.contains(task_name_pattern, na=False))
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
            mask = (data['avg_NPU_runtime [us]'].map(lambda x: math.isnan(x)))
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

    # Calculate averages and geometric means for each common success files by overall and benchmark groups
    for prefix in ["common_success_"]:
        for file in args.files:
            df_with_stats = pd.DataFrame(columns=dfs[prefix + file].columns)
            empty_row_df = pd.DataFrame([[""]*len(dfs[prefix + file].columns)], columns=dfs[prefix + file].columns)
            for group in benchmark_groups:
                if group['benchmark_group_name'] == 'Ungrouped':
                    partial_df = helper_get_ungrouped_df(dfs[prefix + file])
                else:
                    partial_df = helper_get_benchmark_group_df(dfs[prefix + file], group['benchmark_group_patterns'])
                if not partial_df.empty:
                    partial_df_with_stats = augment_stats(partial_df, prefix=group['benchmark_group_name'] + " ")
                    df_with_stats = pd.concat([df_with_stats, partial_df_with_stats, empty_row_df], ignore_index=True)
            if len(benchmark_groups) == 0:
                df_with_stats = helper_get_ungrouped_df(dfs[prefix + file])
            # Add overall stats
            dfs[prefix + file] = augment_stats(dfs[prefix + file], prefix="Overall ")
            dfs[prefix + file] = pd.concat([df_with_stats, empty_row_df, dfs[prefix + file][dfs[prefix + file]["benchmark_name"].str.contains("min|max|average|geometric_mean")]], ignore_index=True)

    # Calculate the normalized values
    for file in args.files:
        dfs["normalized_" + file] = normalize_matching_rows(dfs["common_success_" + baseline_file], dfs["common_success_" + file])

    # Collect overall stats for each file and calculate stats for each group
    dfs["Overall Statistics"] = pd.DataFrame(columns=[
        'Name',
        'Benchmark Group',
        'Total Number Test Cases',
        'Number of Successfully Builded Test Cases',
        'Number of Successfully Compiled Test Cases',
        'Successful Test Cases',
        'Success Rate',
        'Common Successful Test Cases Across All Files',
        'GeoMean Build Time [s]',
        'GeoMean End-to-End Build Time [s]',
        'GeoMean NPU Runtime [us]',
        'GeoMean Total Route Length',
        'Arithmetic Mean Num Net Using Neighbour Sharing',
        'Arithmetic Mean Num Net Using AXI Stream',
        'GeoMean Longest Route Segment',
        'GeoMean Total Buffer Usage [Byte]',
        'GeoMean Average Buffer Usage per Compute Tile [Byte]',
        'GeoMean Average Buffer Usage per Memory Tile [Byte]',
        'Normalized GeoMean Build Time [s]',
        'Normalized GeoMean End-to-End Build Time [s]',
        'Normalized GeoMean NPU Runtime [us]',
        'Normalized GeoMean Total Route Length',
        'Normalized Arithmetic Mean Num Net Using Neighbour Sharing',
        'Normalized Arithmetic Mean Num Net Using AXI Stream',
        'Normalized GeoMean Longest Route Segment',
        'Normalized GeoMean Total Buffer Usage [Byte]',
        'Normalized GeoMean Average Buffer Usage per Compute Tile [Byte]',
        'Normalized GeoMean Average Buffer Usage per Memory Tile [Byte]'
    ])
    empty_row_df = pd.DataFrame([[""]*len(dfs["Overall Statistics"].columns)], columns=dfs["Overall Statistics"].columns)
    for file in args.files:
        total_test_cases = dfs[file].shape[0]
        successful_builded = dfs[file][dfs[file]['build_time [s]'].map(lambda x: not math.isnan(x))].shape[0]
        successful_compiled = dfs[file][dfs[file]['compilation_time [s]'].map(lambda x: not math.isnan(x))].shape[0]
        successful_test_cases = dfs["success_" + file].shape[0]
        common_successful_test_cases = dfs["common_success_" + file][~(dfs["common_success_" + file]["benchmark_name"].str.contains("min|max|average|geometric_mean") | dfs["common_success_" + file]["benchmark_name"].str.match(r"^$"))].shape[0]
        for group in benchmark_groups:
            if group['benchmark_group_name'] == 'Ungrouped':
                gropu_df = helper_get_ungrouped_df(dfs[file])
                group_successful = helper_get_ungrouped_df(dfs["success_" + file]).shape[0]
                group_common_successful = helper_get_ungrouped_df(dfs["common_success_" + file]).shape[0]
            else:
                gropu_df = helper_get_benchmark_group_df(dfs[file], group['benchmark_group_patterns'])
                group_successful = helper_get_benchmark_group_df(dfs["success_" + file], group['benchmark_group_patterns']).shape[0]
                group_common_successful = helper_get_benchmark_group_df(dfs["common_success_" + file], group['benchmark_group_patterns']).shape[0]
            group_total = gropu_df.shape[0]
            group_successful_builded = gropu_df[gropu_df['build_time [s]'].map(lambda x: not math.isnan(x))].shape[0]
            group_successful_compiled = gropu_df[gropu_df['compilation_time [s]'].map(lambda x: not math.isnan(x))].shape[0]
            dfs["Overall Statistics"] = pd.concat([dfs["Overall Statistics"], pd.Series({
                'Name': file_label_map[file],
                'Benchmark Group': group['benchmark_group_name'],
                'Total Number Test Cases': group_total,
                'Number of Successfully Builded Test Cases': group_successful_builded,
                'Number of Successfully Compiled Test Cases': group_successful_compiled,
                'Successful Test Cases': group_successful,
                'Success Rate': float(group_successful) / float(group_total) if float(group_total) > 0.0 else 0.0,
                'Common Successful Test Cases Across All Files': group_common_successful,
                'GeoMean Build Time [s]' : dfs["common_success_" + file][dfs["common_success_" + file]["benchmark_name"] == group['benchmark_group_name'] + " geometric_mean"]['build_time [s]'].values[0] if group_common_successful else '',
                'GeoMean End-to-End Build Time [s]': dfs["common_success_" + file][dfs["common_success_" + file]["benchmark_name"] == group['benchmark_group_name'] + " geometric_mean"]['total_end2end_compilation_time [s]'].values[0] if group_common_successful else '',
                'GeoMean NPU Runtime [us]': dfs["common_success_" + file][dfs["common_success_" + file]["benchmark_name"] == group['benchmark_group_name'] + " geometric_mean"]['avg_NPU_runtime [us]'].values[0] if group_common_successful else '',
                'GeoMean Total Route Length': dfs["common_success_" + file][dfs["common_success_" + file]["benchmark_name"] == group['benchmark_group_name'] + " geometric_mean"]['total_routing_length'].values[0] if group_common_successful else '',
                'Arithmetic Mean Num Net Using Neighbour Sharing': dfs["common_success_" + file][dfs["common_success_" + file]["benchmark_name"] == group['benchmark_group_name'] + " average"]['num_neighbour_sharing_objectFIFO'].values[0] if group_common_successful else '',
                'Arithmetic Mean Num Net Using AXI Stream': dfs["common_success_" + file][dfs["common_success_" + file]["benchmark_name"] == group['benchmark_group_name'] + " average"]['num_circuit_switch_objectFIFO'].values[0] if group_common_successful else '',
                'GeoMean Longest Route Segment': dfs["common_success_" + file][dfs["common_success_" + file]["benchmark_name"] == group['benchmark_group_name'] + " geometric_mean"]['longest_circuit_switch_path'].values[0] if group_common_successful else '',
                'GeoMean Total Buffer Usage [Byte]': dfs["common_success_" + file][dfs["common_success_" + file]["benchmark_name"] == group['benchmark_group_name'] + " geometric_mean"]['total_buffer_size [bytes]'].values[0] if group_common_successful else '',
                'GeoMean Average Buffer Usage per Compute Tile [Byte]': dfs["common_success_" + file][dfs["common_success_" + file]["benchmark_name"] == group['benchmark_group_name'] + " geometric_mean"]['avg_buffer_size_on_compute [bytes]'].values[0] if group_common_successful else '',
                'GeoMean Average Buffer Usage per Memory Tile [Byte]': dfs["common_success_" + file][dfs["common_success_" + file]["benchmark_name"] == group['benchmark_group_name'] + " geometric_mean"]['avg_buffer_size_on_mem [bytes]'].values[0] if group_common_successful else '',
                'Normalized GeoMean Build Time [s]': dfs["normalized_" + file][dfs["normalized_" + file]["benchmark_name"] == group['benchmark_group_name'] + " geometric_mean"]['build_time [s]'].values[0] if group_common_successful else '',
                'Normalized GeoMean End-to-End Build Time [s]': dfs["normalized_" + file][dfs["normalized_" + file]["benchmark_name"] == group['benchmark_group_name'] + " geometric_mean"]['total_end2end_compilation_time [s]'].values[0] if group_common_successful else '',
                'Normalized GeoMean NPU Runtime [us]': dfs["normalized_" + file][dfs["normalized_" + file]["benchmark_name"] == group['benchmark_group_name'] + " geometric_mean"]['avg_NPU_runtime [us]'].values[0] if group_common_successful else '',
                'Normalized GeoMean Total Route Length': dfs["normalized_" + file][dfs["normalized_" + file]["benchmark_name"] == group['benchmark_group_name'] + " geometric_mean"]['total_routing_length'].values[0] if group_common_successful else '',
                'Normalized Arithmetic Mean Num Net Using Neighbour Sharing': dfs["normalized_" + file][dfs["normalized_" + file]["benchmark_name"] == group['benchmark_group_name'] + " average"]['num_neighbour_sharing_objectFIFO'].values[0] if group_common_successful else '',
                'Normalized Arithmetic Mean Num Net Using AXI Stream': dfs["normalized_" + file][dfs["normalized_" + file]["benchmark_name"] == group['benchmark_group_name'] + " average"]['num_circuit_switch_objectFIFO'].values[0] if group_common_successful else '',
                'Normalized GeoMean Longest Route Segment': dfs["normalized_" + file][dfs["normalized_" + file]["benchmark_name"] == group['benchmark_group_name'] + " geometric_mean"]['longest_circuit_switch_path'].values[0] if group_common_successful else '',
                'Normalized GeoMean Total Buffer Usage [Byte]': dfs["normalized_" + file][dfs["normalized_" + file]["benchmark_name"] == group['benchmark_group_name'] + " geometric_mean"]['total_buffer_size [bytes]'].values[0] if group_common_successful else '',
                'Normalized GeoMean Average Buffer Usage per Compute Tile [Byte]': dfs["normalized_" + file][dfs["normalized_" + file]["benchmark_name"] == group['benchmark_group_name'] + " geometric_mean"]['avg_buffer_size_on_compute [bytes]'].values[0] if group_common_successful else '',
                'Normalized GeoMean Average Buffer Usage per Memory Tile [Byte]': dfs["normalized_" + file][dfs["normalized_" + file]["benchmark_name"] == group['benchmark_group_name'] + " geometric_mean"]['avg_buffer_size_on_mem [bytes]'].values[0] if group_common_successful else ''
                }).to_frame().T
            ], ignore_index=True)
        # Add overall stats
        dfs["Overall Statistics"] = pd.concat([dfs["Overall Statistics"], pd.Series({
            'Name': file_label_map[file],
            'Benchmark Group': 'Overall',
            'Total Number Test Cases': total_test_cases,
            'Number of Successfully Builded Test Cases': successful_builded,
            'Number of Successfully Compiled Test Cases': successful_compiled,
            'Successful Test Cases': successful_test_cases,
            'Success Rate': float(successful_test_cases) / float(total_test_cases) if float(total_test_cases) > 0.0 else 0.0,
            'Common Successful Test Cases Across All Files': common_successful_test_cases,
            'GeoMean Build Time [s]' : dfs["common_success_" + file][dfs["common_success_" + file]["benchmark_name"] == "Overall" + " geometric_mean"]['build_time [s]'].values[0] if common_successful_test_cases else '',
            'GeoMean End-to-End Build Time [s]': dfs["common_success_" + file][dfs["common_success_" + file]["benchmark_name"] == "Overall" + " geometric_mean"]['total_end2end_compilation_time [s]'].values[0] if common_successful_test_cases else '',
            'GeoMean NPU Runtime [us]': dfs["common_success_" + file][dfs["common_success_" + file]["benchmark_name"] == "Overall" + " geometric_mean"]['avg_NPU_runtime [us]'].values[0] if common_successful_test_cases else '',
            'GeoMean Total Route Length': dfs["common_success_" + file][dfs["common_success_" + file]["benchmark_name"] == "Overall" + " geometric_mean"]['total_routing_length'].values[0] if common_successful_test_cases else '',
            'Arithmetic Mean Num Net Using Neighbour Sharing': dfs["common_success_" + file][dfs["common_success_" + file]["benchmark_name"] == "Overall" + " average"]['num_neighbour_sharing_objectFIFO'].values[0] if common_successful_test_cases else '',
            'Arithmetic Mean Num Net Using AXI Stream': dfs["common_success_" + file][dfs["common_success_" + file]["benchmark_name"] == "Overall" + " average"]['num_circuit_switch_objectFIFO'].values[0] if common_successful_test_cases else '',
            'GeoMean Longest Route Segment': dfs["common_success_" + file][dfs["common_success_" + file]["benchmark_name"] == "Overall" + " geometric_mean"]['longest_circuit_switch_path'].values[0] if common_successful_test_cases else '',
            'GeoMean Total Buffer Usage [Byte]': dfs["common_success_" + file][dfs["common_success_" + file]["benchmark_name"] == "Overall" + " geometric_mean"]['total_buffer_size [bytes]'].values[0] if common_successful_test_cases else '',
            'GeoMean Average Buffer Usage per Compute Tile [Byte]': dfs["common_success_" + file][dfs["common_success_" + file]["benchmark_name"] == "Overall" + " geometric_mean"]['avg_buffer_size_on_compute [bytes]'].values[0] if common_successful_test_cases else '',
            'GeoMean Average Buffer Usage per Memory Tile [Byte]': dfs["common_success_" + file][dfs["common_success_" + file]["benchmark_name"] == "Overall" + " geometric_mean"]['avg_buffer_size_on_mem [bytes]'].values[0] if common_successful_test_cases else '',
            'Normalized GeoMean Build Time [s]': dfs["normalized_" + file][dfs["normalized_" + file]["benchmark_name"] == "Overall" + " geometric_mean"]['build_time [s]'].values[0] if common_successful_test_cases else '',
            'Normalized GeoMean End-to-End Build Time [s]': dfs["normalized_" + file][dfs["normalized_" + file]["benchmark_name"] == "Overall" + " geometric_mean"]['total_end2end_compilation_time [s]'].values[0] if common_successful_test_cases else '',
            'Normalized GeoMean NPU Runtime [us]': dfs["normalized_" + file][dfs["normalized_" + file]["benchmark_name"] == "Overall" + " geometric_mean"]['avg_NPU_runtime [us]'].values[0] if common_successful_test_cases else '',
            'Normalized GeoMean Total Route Length': dfs["normalized_" + file][dfs["normalized_" + file]["benchmark_name"] == "Overall" + " geometric_mean"]['total_routing_length'].values[0] if common_successful_test_cases else '',
            'Normalized Arithmetic Mean Num Net Using Neighbour Sharing': dfs["normalized_" + file][dfs["normalized_" + file]["benchmark_name"] == "Overall" + " average"]['num_neighbour_sharing_objectFIFO'].values[0] if common_successful_test_cases else '',
            'Normalized Arithmetic Mean Num Net Using AXI Stream': dfs["normalized_" + file][dfs["normalized_" + file]["benchmark_name"] == "Overall" + " average"]['num_circuit_switch_objectFIFO'].values[0] if common_successful_test_cases else '',
            'Normalized GeoMean Longest Route Segment': dfs["normalized_" + file][dfs["normalized_" + file]["benchmark_name"] == "Overall" + " geometric_mean"]['longest_circuit_switch_path'].values[0] if common_successful_test_cases else '',
            'Normalized GeoMean Total Buffer Usage [Byte]': dfs["normalized_" + file][dfs["normalized_" + file]["benchmark_name"] == "Overall" + " geometric_mean"]['total_buffer_size [bytes]'].values[0] if common_successful_test_cases else '',
            'Normalized GeoMean Average Buffer Usage per Compute Tile [Byte]': dfs["normalized_" + file][dfs["normalized_" + file]["benchmark_name"] == "Overall" + " geometric_mean"]['avg_buffer_size_on_compute [bytes]'].values[0] if common_successful_test_cases else '',
            'Normalized GeoMean Average Buffer Usage per Memory Tile [Byte]': dfs["normalized_" + file][dfs["normalized_" + file]["benchmark_name"] == "Overall" + " geometric_mean"]['avg_buffer_size_on_mem [bytes]'].values[0] if common_successful_test_cases else ''
            }).to_frame().T,
            empty_row_df
        ], ignore_index=True)

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

        # Add border to the min, max, average and geometric_mean rows in each sheet with orange border
        enclosed_row_fields = ["min", "max", "average", "geometric_mean"]
        num_enclosed_rows = len(enclosed_row_fields)
        def add_border_to_enclosed_rows(worksheet):
            thick_border_side = Side(border_style="thick", color="FFA500") # orange border
            for row in range(2, worksheet.max_row + 1):
                if any(item in worksheet.cell(row=row, column=2).value for item in enclosed_row_fields):
                    for col in range(1, worksheet.max_column + 1):
                        # only add border to the top and bottom of the enclosed rows
                        # only add left and right border to leftmost and rightmost columns
                        top_side = Side(border_style="none")
                        bottom_side = Side(border_style="none")
                        left_side = Side(border_style="none")
                        right_side = Side(border_style="none")
                        if (worksheet.cell(row=row - 1, column=2).value is None) or (not any(item in worksheet.cell(row=row - 1, column=2).value for item in enclosed_row_fields)):
                            top_side = thick_border_side
                        if (worksheet.cell(row=row + 1, column=2).value is None) or (not any(item in worksheet.cell(row=row + 1, column=2).value for item in enclosed_row_fields)):
                            bottom_side = thick_border_side
                        if col == 1:
                            left_side = thick_border_side
                        if col == worksheet.max_column:
                            right_side = thick_border_side
                        worksheet.cell(row=row, column=col).border = Border(
                            left=left_side,
                            right=right_side,
                            top=top_side,
                            bottom=bottom_side
                        )

        for file in args.files:
            file_name = file_label_map[file]
            baseline_file_name = file_label_map[baseline_file]
            worksheet = writer.sheets[f"common_success_{file_name}"[:31]]
            add_border_to_enclosed_rows(worksheet)

            worksheet = writer.sheets[f"normalized_{file_name}_vs_{baseline_file_name}"[:31]]
            add_border_to_enclosed_rows(worksheet)
        
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
