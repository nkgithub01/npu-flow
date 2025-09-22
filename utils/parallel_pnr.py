#!/usr/bin/env python3
"""
This script finds all <some_task>.placed_netlist.pnr.json files in input_dir/<some_benchmark>
subdirectories and places-and-routes them in parallel. It creates output files,
route summary files, and detailed log files for each processed file.

Usage:
    python parallel_pnr.py input_dir -o output_dir [-j jobs]
"""

import argparse
import os
import glob
import subprocess
import time
import traceback
from multiprocessing import Pool
import sys


def log_heading(heading, char="-"):
    return f"{char * 40}\n{heading}\n{char * 40}"


def find_pnr_files(input_dir):
    pattern = os.path.join(input_dir, "*", "*.placed_netlist.pnr.json")
    files = glob.glob(pattern)
    return sorted(files)  # Sort for consistent ordering


def process_single_file(job_args):
    input_file, output_file, log_file, pnr_args = job_args
    # TODO: remove capturing route summary via PnR tool in future versions
    output_route_summary = output_file.replace(
        ".placed_netlist.pnr.json", ".route_summary.pnr.json"
    )

    # Result dictionary to capture execution details
    r = {
        "input_file": input_file,
        "output_file": output_file,
        "output_route_summary": output_route_summary,
        "log_file": log_file,
        "start_time": time.time(),
    }

    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Run the placer command
        pnr_bin = os.path.expandvars("$NPU_PNR_BIN_DIR/placer")
        assert os.path.exists(pnr_bin), f"PnR binary not found at {pnr_bin}"

        cmd = [
            pnr_bin,
            input_file,
            "--output",
            output_file,
            "--route-summary",
            output_route_summary,
        ]
        pnr_args = pnr_args.strip()
        if pnr_args:
            cmd.extend(pnr_args.split())
        r["cmd"] = " ".join(cmd)

        print(f"Processing: {os.path.basename(input_file)}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=None)
        r["returncode"] = result.returncode
        r["stdout"] = result.stdout.strip()
        r["stderr"] = result.stderr.strip()

    except Exception as e:
        r["python_exception"] = {
            "exception_type": type(e).__name__,
            "exception_message": str(e),
            "full_traceback": traceback.format_exc(),
        }

    r["end_time"] = time.time()
    r["execution_time"] = r["end_time"] - r["start_time"]

    with open(log_file, "w") as f:
        f.write(log_heading("PnR Log") + "\n")
        f.write(
            "Input file:     {input_file}\n"
            "Output file:    {output_file}\n"
            "Route summary:  {output_route_summary}\n"
            "Command:        {cmd}\n"
            "Execution time: {execution_time:.2f} secs\n".format(**r)
        )

        if "python_exception" in r:
            f.write(f"Status:         FAILED (Python Exception)")
            r["success"] = False
        elif r["returncode"] != 0:
            f.write(f"Status:         FAILED (Return code {r["returncode"]})")
            r["success"] = False
        else:
            f.write(f"Status:         SUCCESS")
            r["success"] = True

        f.write("\n\n" + log_heading("Stdout") + "\n")
        if "stdout" in r and r["stdout"]:
            f.write(r["stdout"])
        else:
            f.write("(no stdout output)")

        f.write("\n\n" + log_heading("Stderr") + "\n")
        if "stderr" in r and r["stderr"]:
            f.write(r["stderr"])
        else:
            f.write("(no stderr output)")

        f.write("\n\n" + log_heading("Python Exception") + "\n")
        if "python_exception" in r:
            msg = r["python_exception"]
            f.write(f"Exception Type: {msg['exception_type']}\n")
            f.write(f"Message:        {msg['exception_message']}\n")
            f.write(f"Traceback:\n{msg["full_traceback"]}")
        else:
            f.write("(no python exception)\n")

    return r


def main():
    parser = argparse.ArgumentParser(
        description="Run PnR stage in parallel with placement files from input directory",
    )

    parser.add_argument(
        "input_dir",
        help="Input directory containing benchmark subdirectories with *.placed_netlist.pnr.json files",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Output directory for processed files and logs",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,  # multiprocessing.Pool uses number of cores if not specified
        help="Number of parallel jobs (default: number of CPU cores)",
    )
    parser.add_argument(
        "--pnr-args",
        type=str,
        default="",
        help="Additional arguments to pass to the PnR tool",
    )

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    # Validate input directory
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist")
        sys.exit(1)

    # Find all PnR files
    print(f"Searching for <some_task>.placed_netlist.pnr.json files in {input_dir} ...")
    pnr_files = find_pnr_files(input_dir)

    if not pnr_files:
        print(f"No <some_task>.placed_netlist.pnr.json files found in {input_dir}")
        print(
            "Make sure your directory structure is: "
            "input_dir/<some_benchmark>/<some_task>.placed_netlist.pnr.json"
        )
        sys.exit(1)

    print(f"Found {len(pnr_files)} PnR files to process:")
    for f in pnr_files:
        rel_path = os.path.relpath(f, input_dir)
        print(f"    {rel_path}")

    # Prepare thread/job arguments for processing
    job_args = []
    for input_file in pnr_files:
        # Extract relative path from input_dir
        rel_path = os.path.relpath(input_file, input_dir)

        # Create output file path (same name and extension)
        output_file = os.path.join(output_dir, rel_path)

        # Create log file path (same task/base name but with .pnr.log extension)
        log_file = output_file.replace(".placed_netlist.pnr.json", ".pnr.log")

        job_args.append((input_file, output_file, log_file, args.pnr_args))

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nStarting parallel processing with {args.jobs or 'auto'} jobs...")
    print(f"Output directory: {output_dir}")

    start_total = time.time()

    # Process files in parallel
    try:
        with Pool(processes=args.jobs) as pool:
            results = pool.map(process_single_file, job_args)
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)

    end_total = time.time()
    total_wall_time = end_total - start_total

    # Print summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    total_cpu_time = sum(r["execution_time"] for r in results)

    print("\n" + log_heading("Parallel PnR Summary"))
    print(f"Total files processed: {len(results)}")
    print(f"Successful:            {successful}")
    print(f"Failed:                {failed}")
    print(f"Total wall time:       {total_wall_time:.2f} seconds")
    print(f"Total CPU time:        {total_cpu_time:.2f} seconds")

    if successful > 0:
        print(f"\nSuccessful files:")
        for r in results:
            if r["success"]:
                rel_input = os.path.relpath(r["input_file"], input_dir)
                print(f"    {rel_input} ({r['execution_time']:.2f}s)")

    if failed > 0:
        print(f"\nFailed files:")
        for r in results:
            if not r["success"]:
                rel_input = os.path.relpath(r["input_file"], input_dir)
                error_info = r.get(
                    "error", f"Return code {r.get('return_code', 'unknown')}"
                )
                print(f"    {rel_input}: {error_info}")
                print(f"    Log: {r['log_file']}")

    print(f"\nAll log files are saved in: {output_dir}")

    # Exit with error code if any files failed
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
