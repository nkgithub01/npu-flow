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
    shim_pid_to_name = {
        entry["pid"]: entry.get("args", {}).get("name", "")
        for entry in process_data
        if "shim" in entry.get("args", {}).get("name", "")
    }

    # Sort process entries by timestamp
    event_data = [entry for entry in data if entry.get("name") != "process_name" and "ts" in entry]
    sorted_event = sorted(event_data, key=lambda x: x["ts"])

    # Measure time deltas
    event_delta = {}
    for pid, shim_name in shim_pid_to_name.items():
        pid_data = [entry for entry in sorted_event if entry.get("pid") == pid]
        start_time = None
        for entry in pid_data:
            if entry["name"] == "DMA_MM2S_0_START_TASK" and entry["ph"] == "B":
                start_time = entry["ts"]
            elif entry["name"] == "DMA_S2MM_0_FINISHED_TASK" and entry["ph"] == "B":
                if start_time is None:
                    raise Exception("S2MM_FINISHED_TASK without MM2S_START_TASK")
                delta = entry["ts"] - start_time
                event_delta.setdefault(shim_name, []).append(delta)
                start_time = None

    # Collect and print results
    output_lines = []
    for name, diffs in event_delta.items():
        trimmed = diffs[args.warmup:]
        if trimmed:
            line = f"{name} | Min: {min(trimmed)} cycles, Max: {max(trimmed)} cycles, Avg: {sum(trimmed) / len(trimmed):.2f} cycles"
            print(line)
            output_lines.append(line)

    # Write to output file if specified
    if args.output:
        with open(args.output, 'w') as f:
            for line in output_lines:
                f.write(line + "\n")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze shim DMA latencies from trace JSON.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the trace JSON file")
    parser.add_argument("-w", "--warmup", type=int, default=0, help="Number of warm-up iterations to skip")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to output summary txt file")
    args = parser.parse_args()
    main(args)
