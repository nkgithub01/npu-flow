import json
import argparse

def main(args):
    # Load JSON data
    with open(args.input, 'r') as f:
        data = json.load(f)
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            print(f"Entry {i} is not a dict: {type(entry)}, value: {entry}")
    # Separate process name entries
    process_data = [entry for entry in data if entry.get("name") == "process_name"]
    
    # Map shim PID to tile name
    shim_pids = {
        entry["pid"]: entry.get("args", {}).get("name", "")
        for entry in process_data
        if "shim" in entry.get("args", {}).get("name", "")
    }

    # Sort pid process entries by timestamp
    event_data = [entry for entry in data if entry.get("name") != "process_name" and "ts" in entry]
    sorted_event = sorted(event_data, key=lambda x: x["ts"])

    # Extract earliest START_TASK and latest FINISHED_TASK per shim_pid
    dma_event_map = {pid: [] for pid in shim_pids.keys()}
    for shim_pid in shim_pids.keys():
        pid_data = [entry for entry in sorted_event if entry.get("pid") == shim_pid]

        start_times = [
            entry["ts"]
            for entry in pid_data
            if entry["name"] == "DMA_MM2S_0_START_TASK" and entry["ph"] == "B"
        ]
        end_times = [
            entry["ts"]
            for entry in pid_data
            if entry["name"] == "DMA_S2MM_0_FINISHED_TASK" and entry["ph"] == "B"
        ]

        if not start_times or not end_times:
            raise Exception(
                f"Missing START_TASK or FINISHED_TASK for pid {shim_pid}. "
                "Trace size might not be large enough."
            )

        # Store tuple of (earliest_start, latest_end)
        dma_event_map[shim_pid].append((min(start_times), max(end_times)))


    lengths = [len(lst) for lst in dma_event_map.values()]
    assert all(l == lengths[0] for l in lengths), (
        "Number of DMA events do not match between Shims. "
        "Trace size might not be large enough."
    )
    # Calculate cycle duration per iteration
    output = []
    for i in range(lengths[0]):
        ith_dma_event = [dma_event_map[pid][i] for pid in dma_event_map]
        min_start = min(start for start, _ in ith_dma_event)
        max_end = max(end for _, end in ith_dma_event)
        output.append((min_start,max_end,max_end - min_start))

    if args.output: 
        with open(args.output, 'w') as f:
            f.write("iter,start_cycle_num,end_cycle_num,duration_in_cycles\n")
            for i, line in enumerate(output):
                f.write(f"{i},{line[0]},{line[1]},{line[2]}\n")
            durations = [line[2] for line in output]
            summary = (
                "\n======== Cycle Summary ==========\n"
                f"Total iterations: {len(durations)}\n"
                f"Min     : {min(durations)} cycles\n"
                f"Max     : {max(durations)} cycles\n"
                f"Average : {sum(durations) / len(durations):.2f} cycles\n"
            )
            f.write(summary)
            print(summary)        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze shim DMA latencies from trace JSON.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the trace JSON file")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to output summary txt file")
    args = parser.parse_args()
    main(args)
