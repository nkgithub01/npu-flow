#!/usr/bin/env python3
"""
Compare Results Script

This script compares multiple CSV files and creates an Excel output with normalized values.
The normalization is performed against the first input CSV file as the baseline.
When calculating normalized values, benchmark and task names must match.
Missing values are filled with "N/A".
"""

import pandas as pd
import sys
import argparse
from pathlib import Path


def load_csv_file(file_path):
    """Load CSV file and return DataFrame with proper indexing."""
    try:
        df = pd.read_csv(file_path)
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        # Strip whitespace from string columns
        for col in ['benchmark', 'task_name']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def create_key_from_row(row):
    """Create a unique key from benchmark and task_name."""
    return f"{row['benchmark']}_{row['task_name']}"


def normalize_dataframes(baseline_df, comparison_df):
    """Normalize comparison_df against baseline_df."""
    # Create dictionary for fast lookup of comparison data using benchmark_task as key
    comparison_dict = {}
    for _, row in comparison_df.iterrows():
        key = create_key_from_row(row)
        comparison_dict[key] = row
    
    # Prepare result data
    result_data = []
    
    # Get numeric columns for normalization (exclude benchmark and task_name)
    numeric_columns = [col for col in baseline_df.columns 
                      if col not in ['benchmark', 'task_name'] and 
                      baseline_df[col].dtype in ['int64', 'float64']]
    
    # Iterate through baseline rows in order to maintain baseline ordering
    for _, baseline_row in baseline_df.iterrows():
        key = create_key_from_row(baseline_row)
        comparison_row = comparison_dict.get(key)
        
        # Extract benchmark and task_name from baseline row
        benchmark = baseline_row['benchmark']
        task_name = baseline_row['task_name']
        
        row_data = {
            'benchmark': benchmark,
            'task_name': task_name
        }
        
        # Add only normalized values for each numeric column
        for col in numeric_columns:
            baseline_val = None
            comparison_val = None
            
            if baseline_row is not None:
                try:
                    baseline_val = float(baseline_row[col]) if pd.notna(baseline_row[col]) and str(baseline_row[col]).upper() != 'N/A' else None
                except (ValueError, TypeError):
                    baseline_val = None
            
            if comparison_row is not None:
                try:
                    comparison_val = float(comparison_row[col]) if pd.notna(comparison_row[col]) and str(comparison_row[col]).upper() != 'N/A' else None
                except (ValueError, TypeError):
                    comparison_val = None
            
            # Calculate normalized value (comparison / baseline)
            if baseline_val is not None and comparison_val is not None and baseline_val != 0:
                normalized_val = comparison_val / baseline_val
                row_data[col] = normalized_val
            else:
                row_data[col] = 'N/A'
        
        result_data.append(row_data)
    
    # Add any comparison rows that don't have a matching baseline entry
    baseline_keys = set(create_key_from_row(row) for _, row in baseline_df.iterrows())
    for _, comparison_row in comparison_df.iterrows():
        key = create_key_from_row(comparison_row)
        if key not in baseline_keys:
            # This comparison row has no matching baseline
            benchmark = comparison_row['benchmark']
            task_name = comparison_row['task_name']
            
            row_data = {
                'benchmark': benchmark,
                'task_name': task_name
            }
            
            # All values will be N/A since there's no baseline to compare against
            for col in numeric_columns:
                row_data[col] = 'N/A'
            
            result_data.append(row_data)
    
    return pd.DataFrame(result_data)


def create_excel_output(baseline_df, comparison_dfs, output_file, csv_names):
    """Create Excel file with multiple sheets containing original and normalized values."""
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Write baseline sheet
        baseline_df.to_excel(writer, sheet_name=f'Baseline_{csv_names[0]}', index=False)
        
        # Write comparison sheets with normalized values
        for i, (comparison_df, csv_name) in enumerate(zip(comparison_dfs, csv_names[1:]), 1):
            # Create normalized comparison
            normalized_df = normalize_dataframes(baseline_df, comparison_df)
            
            # Write to sheet
            sheet_name = f'Comparison_{csv_name}'
            # Limit sheet name to Excel's 31 character limit
            if len(sheet_name) > 31:
                sheet_name = f'Comp_{i}_{csv_name}'[:31]
            
            normalized_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Also write the raw comparison data
            raw_sheet_name = f'Raw_{csv_name}'
            if len(raw_sheet_name) > 31:
                raw_sheet_name = f'Raw_{i}_{csv_name}'[:31]
            comparison_df.to_excel(writer, sheet_name=raw_sheet_name, index=False)


def main():
    parser = argparse.ArgumentParser(
        description='Compare CSV benchmark results and create Excel output with normalized values',
        epilog='''Examples:
  %(prog)s baseline.csv comparison1.csv comparison2.csv -o results.xlsx
  %(prog)s placed.csv unplaced1.csv unplaced2.csv unplaced3.csv
  
  The first CSV file is always used as the baseline for normalization.
  Each additional CSV file will be compared against the baseline.''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('csv_files', nargs='+', 
                       help='Input CSV files (first file is used as baseline, supports 2 or more files)')
    parser.add_argument('-o', '--output', default='comparison_results.xlsx', 
                       help='Output Excel file (default: comparison_results.xlsx)')
    
    args = parser.parse_args()
    
    if len(args.csv_files) < 2:
        print("Error: At least 2 CSV files are required for comparison")
        sys.exit(1)
    
    # Load all CSV files
    dataframes = []
    csv_names = []
    
    for csv_file in args.csv_files:
        file_path = Path(csv_file)
        if not file_path.exists():
            print(f"Error: File {csv_file} does not exist")
            sys.exit(1)
        
        df = load_csv_file(csv_file)
        if df is None:
            sys.exit(1)
        
        dataframes.append(df)
        csv_names.append(file_path.stem)
        print(f"Loaded {len(df)} rows from {csv_file}")
    
    # Use first dataframe as baseline
    baseline_df = dataframes[0]
    comparison_dfs = dataframes[1:]
    
    # Create Excel output
    print(f"Creating Excel output: {args.output}")
    create_excel_output(baseline_df, comparison_dfs, args.output, csv_names)
    
    print(f"Comparison complete! Results saved to {args.output}")
    
    # Print summary
    print("\nSummary:")
    print(f"Baseline: {csv_names[0]} ({len(baseline_df)} rows)")
    for i, (df, name) in enumerate(zip(comparison_dfs, csv_names[1:]), 1):
        print(f"Comparison {i}: {name} ({len(df)} rows)")


if __name__ == "__main__":
    # If run without arguments, use the default CSV files in the workspace
    if len(sys.argv) == 1:
        # Default behavior: compare the two CSV files in the workspace
        baseline_file = "results_placed_level_IRON.csv"
        comparison_file = "results_unplaced_level_IRON.csv"
        output_file = "comparison_results.xlsx"
        
        if Path(baseline_file).exists() and Path(comparison_file).exists():
            print("No arguments provided. Using default files:")
            print(f"Baseline: {baseline_file}")
            print(f"Comparison: {comparison_file}")
            print(f"Output: {output_file}")
            
            # Load the files
            baseline_df = load_csv_file(baseline_file)
            comparison_df = load_csv_file(comparison_file)
            
            if baseline_df is not None and comparison_df is not None:
                csv_names = ["placed_level_IRON", "unplaced_level_IRON"]
                create_excel_output(baseline_df, [comparison_df], output_file, csv_names)
                print(f"Comparison complete! Results saved to {output_file}")
                
                # Print summary
                print("\nSummary:")
                print(f"Baseline: {csv_names[0]} ({len(baseline_df)} rows)")
                print(f"Comparison: {csv_names[1]} ({len(comparison_df)} rows)")
            else:
                print("Error loading default files")
                sys.exit(1)
        else:
            print("Default CSV files not found. Please provide CSV file paths as arguments.")
            print("Usage: python compare_results.py baseline.csv comparison1.csv [comparison2.csv ...] [-o output.xlsx]")
            print("\nExamples:")
            print("  python compare_results.py placed.csv unplaced.csv")
            print("  python compare_results.py baseline.csv test1.csv test2.csv test3.csv -o results.xlsx")
            sys.exit(1)
    else:
        main()
