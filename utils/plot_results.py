import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import pandas as pd
import argparse

def main(args):
    ##########################################################################################
    #
    # Plot number of successfully solved benchmarks for different placement methods
    #
    ##########################################################################################
    # Read from Excel
    excel_path = args.results_across_all_algorithms_excel_file
    df_success = pd.read_excel(excel_path, sheet_name="Overall Statistics")

    # Data
    categories = [
        "Small Pipeline\nSynthetic",
        "Large Pipeline\nSynthetic",
        "Small Feedback Loop\nSynthetic",
        "Large Feedback Loop\nSynthetic",
        "Small\nReal World Application",
        "Large\nReal World Application",
        "Total"
    ]
    num_testcases = df_success.loc[:len(categories)-1, "Total Number Test Cases"].astype(int).tolist()

    methods = {
        "Manual Placement" :    round(df_success.loc[df_success["Name"] == "HP_PNR",    "Success Rate"]*100).astype(int).tolist(),
        "AMD Placer" :          round(df_success.loc[df_success["Name"] == "SP",        "Success Rate"]*100).astype(int).tolist(),
        "SA + BB" :             round(df_success.loc[df_success["Name"] == "SA",        "Success Rate"]*100).astype(int).tolist(),
        "SA + BBCG" :           round(df_success.loc[df_success["Name"] == "SA_BBCG",   "Success Rate"]*100).astype(int).tolist(),
        "SA + MILP" :           round(df_success.loc[df_success["Name"] == "SA_MILP",   "Success Rate"]*100).astype(int).tolist(),
        "LSMO" :                round(df_success.loc[df_success["Name"] == "LSMO",      "Success Rate"]*100).astype(int).tolist(),
        "MILP" :                round(df_success.loc[df_success["Name"] == "MILP",      "Success Rate"]*100).astype(int).tolist()
    }
    # Remove methods with incorrect data length
    methods = {k: v for k, v in methods.items() if len(v) == len(categories)}

    # Create combined x-axis labels
    x_labels = [
        f"{cat}\n(N={n})"
        for cat, n in zip(categories, num_testcases)
    ]

    # Plot settings
    x = np.arange(len(categories)) * (len(methods)+1)
    bar_width = 0.9
    fig, ax = plt.subplots(figsize=(16, 4))

    # Plot bars
    multiplier = 0
    for algorithm, success_rate in methods.items():
        offset = bar_width * multiplier
        rects = ax.bar(x + offset, success_rate, bar_width, label=algorithm)
        ax.bar_label(rects, padding=3, rotation=55, fontsize=12)
        multiplier += 1

    plt.xlabel("Benchmark Categories", fontsize=14, fontweight='bold')
    plt.ylabel("Successfully Solved \nBenchmarks (%)", fontsize=14, fontweight='bold')
    plt.xticks(
        x + bar_width * (len(methods) - 1) / 2,
        x_labels,
        rotation=0,
        ha="center",
        fontsize=12
    )
    plt.yticks(np.arange(0, 130, 25), fontsize=14)

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=len(methods), fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()

    # Save as SVG
    plt.savefig(args.output_dir + "/successfully_solved_testcase_in_each_Benchmark_group.svg", format="svg", bbox_inches='tight')


    ##########################################################################################
    #
    # Plot QoR metrics comparison between SA Placer and LSMO
    #
    ##########################################################################################
    # Read from Excel
    excel_path = args.results_SA_vs_LSMO_excel_file
    df_success = pd.read_excel(excel_path, sheet_name="Overall Statistics")

    # Data
    categories = [
        "NPU Runtime",
        "Total\nRoute Length",
        "Num Net\nUsing\nShared Memory",
        "Num Net\nUsing\nAXI Stream",
        "Average\nAXI Stream\nRoute Length",
        "Total\nBuffer Usage",
        "Average\nBuffer Usage\nper Compute Tile"
    ]
    fields = [
        "Normalized GeoMean NPU Runtime [us]",
        "Normalized GeoMean Total Route Length",
        "Normalized Arithmetic Mean Num Net Using Neighbour Sharing",
        "Normalized Arithmetic Mean Num Net Using AXI Stream",
        "Normalized GeoMean Average AXI Route Length",
        "Normalized GeoMean Total Buffer Usage [Byte]",
        "Normalized GeoMean Average Buffer Usage per Compute Tile [Byte]"
    ]

    methods = {
        "Manual Placement" :    df_success.loc[(df_success["Name"] == "HP_PNR") & (df_success["Benchmark Group"] == "Overall"), fields].astype(float).iloc[0].tolist(),
        "SA + BBCG" :           df_success.loc[(df_success["Name"] == "SA_BBCG") & (df_success["Benchmark Group"] == "Overall"), fields].astype(float).iloc[0].tolist(),
        "LSMO" :                df_success.loc[(df_success["Name"] == "LSMO") & (df_success["Benchmark Group"] == "Overall"), fields].astype(float).iloc[0].tolist()
    }
    # Remove methods with incorrect data length
    methods = {k: v for k, v in methods.items() if len(v) == len(categories)}

    # Plot settings
    x = np.arange(len(categories)) * (len(methods)+1)
    bar_width = 1
    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot bars
    multiplier = 0
    for algorithm, success_rate in methods.items():
        offset = bar_width * multiplier
        rects = ax.bar(x + offset, success_rate, bar_width, label=algorithm)
        ax.bar_label(rects, padding=3, rotation=90, fontsize=25, fmt=lambda x: f'{x:.2f}'.rstrip('0').rstrip('.'))
        multiplier += 1

    plt.xlabel("Benchmark Categories", fontsize=25, fontweight='bold')
    plt.ylabel("Metric Value\n(Normalized)", fontsize=25, fontweight='bold')
    plt.xticks(
        x + bar_width * (len(methods) - 1) / 2,
        categories,
        rotation=0,
        ha="center",
        fontsize=18
    )
    plt.ylim(0, 1.6)
    plt.yticks(np.arange(0, 1.6, 0.25), fontsize=25)

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=len(methods), fontsize=25)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()

    # Save as SVG
    plt.savefig(args.output_dir + "/placement_qor_metrics_comparison_between_SAPlacer_and_LSMO.svg", format="svg", bbox_inches='tight')


    ##########################################################################################
    #
    # Plot runtime vs netlist size comparison: Manual Placement vs algorithms (scatter plot)
    #
    ##########################################################################################
    # Read from Excel
    excel_path = args.results_across_all_algorithms_excel_file
    algorithm_labels = {
        "AMD Placer": "SP",
        "SA + BB": "SA",
        "SA + BBCG": "SA_BBCG",
        "SA + MILP": "SA_MILP",
        "LSMO": "LSMO",
        "MILP": "MILP"
    }
    algorithms = list(algorithm_labels.keys())
    df = dict()
    for algorithm, sheet_name in algorithm_labels.items():
        try:
            df[algorithm] = pd.read_excel(excel_path, sheet_name=sheet_name, thousands=",")
        except Exception as e:
            print(f"Warning: {algorithm} data: {e}")
            algorithms.remove(algorithm)

    # Columns
    x_col = "netlist_size"
    y_col = "build_time [s]"

    # Create figure
    plt.figure(figsize=(12, 8))
    style_map = {
        "AMD Placer": {
            "color": "#ff0000", # Red
            "marker": "o",        # Cross
            "size": 40
        },
        "SA + BB": {
            "color": "#0004FF", # Blue
            "marker": "^",        # Triangle
            "size": 40
        },
        "SA + BBCG": {
            "color": "#ff00ff", # Pink
            "marker": "v",        # Triangle
            "size": 40
        },
        "SA + MILP": {
            "color": "#006FA6", # Purple
            "marker": "^",        # Triangle
            "size": 40
        },
        "LSMO": {
            "color": "#2ca02c", # Green
            "marker": "*",        # Star
            "size": 80
        },
        "MILP": {
            "color": "#dd7700", # Orange
            "marker": "s",        # Square
            "size": 40
        },
    }


    # Plot scatter + trendline for each algorithm
    r2_values = []
    def r2_score(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
    for algorithm in algorithms:
        data = df[algorithm][[x_col, y_col]].dropna()

        x = data[x_col].values
        y = data[y_col].values

        mask = x <= 90
        x = x[mask]
        y = y[mask]

        # Scatter
        plt.scatter(
            x,
            y,
            s=style_map[algorithm]["size"],
            marker=style_map[algorithm]["marker"],
            color=style_map[algorithm]["color"],
            edgecolors="none",
            linewidths=1,
            alpha=1,
            label=algorithm
        )

        # --- Trendline ---
        if algorithm in ["AMD Placer"]:
            # Polynomial fit (degree 1)
            coeffs, residuals, rank, singular_values, rcond = np.polyfit(x, y, 1, full=True)

            x_min = x.min() * 0.7
            x_max = x.max() * 1.3
            trend_x = np.linspace(x_min, x_max, 300)
            trend_y = np.polyval(coeffs, trend_x)

            # Avoid negative values (important for visual correctness)
            trend_y[trend_y < 1e-3] = np.nan

            # Calculate R² using the residuals from the polynomial fit
            y_pred = np.polyval(coeffs, x)
            r2_values.append(r2_score(y, y_pred))

        elif algorithm in ["SA + MILP"]:
            # Polynomial fit (degree 2)
            coeffs, residuals, rank, singular_values, rcond = np.polyfit(x, y, 2, full=True)

            x_min = x.min() * 0.7
            x_max = x.max() * 1.3
            trend_x = np.linspace(x_min, x_max, 300)
            trend_y = np.polyval(coeffs, trend_x)

            # Avoid negative values (important for visual correctness)
            trend_y[trend_y < 1e-3] = np.nan

            # Calculate R² using the residuals from the polynomial fit
            y_pred = np.polyval(coeffs, x)
            r2_values.append(r2_score(y, y_pred))

        else:
            # Power-law fit (log-log)
            log_x = np.log10(x)
            log_y = np.log10(y)

            coeffs, residuals, rank, singular_values, rcond = np.polyfit(log_x, log_y, 1, full=True)

            log_x_min = log_x.min() - 0.5
            log_x_max = log_x.max() + 0.4

            trend_x = np.logspace(log_x_min, log_x_max, 300)
            trend_y = (10 ** coeffs[1]) * (trend_x ** coeffs[0])

            # Calculate R² using the residuals from the polynomial fit
            log_y_pred = np.polyval(coeffs, log_x)
            y_pred = 10 ** log_y_pred
            r2_values.append(r2_score(y, y_pred))

        plt.plot(
            trend_x,
            trend_y,
            color=style_map[algorithm]["color"],
            linewidth=3,
            alpha=0.75
        )



    # Labels
    plt.xlabel("Netlist Size", fontsize=20, fontweight='bold')
    plt.xlim(0, 90)
    plt.xticks(np.arange(0, 95, 10), fontsize=18)
    plt.ylabel("Placement Solving Time [s]", fontsize=20, fontweight='bold')
    plt.ylim(0, 4500)
    plt.yticks(np.arange(0, 4500, 1000), fontsize=18)

    # Grid
    plt.grid(True, which="both", linewidth=1, alpha=0.6, linestyle="--")

    # Legend
    # Create custom legend labels with R² values
    legend_labels = []
    for algorithm, r2 in zip(algorithms, r2_values):
        legend_labels.append(f"{algorithm}")
        legend_labels.append(f"R²={r2:.2f}")
    plt.legend(legend_labels, fontsize=14, loc="upper center", ncol=len(algorithms), bbox_to_anchor=(0.44, 1.12))

    # Ticks
    # plt.tick_params(axis="both", which="major", labelsize=13)

    # Layout
    plt.tight_layout()
    plt.savefig(args.output_dir + "/placement_solving_time_vs_netlist_size_legend.svg", format="svg", bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting script for placement results")
    parser.add_argument(
        "--results_across_all_algorithms_excel_file",
        type=str,
        default="Results_across_all_algorithms.xlsx",
        help="Path to the input Excel file containing the results across all algorithms"
    )
    parser.add_argument(
        "--results_SA_vs_LSMO_excel_file",
        type=str,
        default="Results_SA_vs_LSMO.xlsx",
        help="Path to the input Excel file containing the results for SA + BBCG vs LSMO comparison"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save the output plots"
    )
    args = parser.parse_args()
    main(args)