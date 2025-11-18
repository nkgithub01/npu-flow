import sys
import re
import json
import os
import glob
import argparse

# prefix components:
prt_space =  '    '
prt_branch = '│   '
# pointers:
prt_tee =    '├── '
prt_last =   '└── '

def parse_mlir_file(mlir_file_path, result_file, verbose=False):
    fifo_count = 0
    unicast_fifo_count = 0
    multicast_fifo_count = 0
    link_count = 0

    with open(mlir_file_path, 'r') as f:
        for line in f:
            # Count objectFIFO ops "aie.objectfifo"
            if re.search(r'\saie\.objectfifo\s', line): 
                fifo_count += 1
                # Check if this objectFIFO is unicast or multicast
                match = re.search(r'\{([^\}]*)\}', line)
                if match:
                    items = [item.strip() for item in match.group(1).split(',') if item.strip()]
                    if len(items) == 1:
                        unicast_fifo_count += 1
                    elif len(items) > 1:
                        multicast_fifo_count += 1
            # Count objectFIFO link ops (e.g., "aie.objectfifo.link" or similar)
            if re.search(r'\saie\.objectfifo\.link\s', line):
                link_count += 1

    if verbose:
        print(f"Number of objectFIFO: {fifo_count}")
        print(f"{prt_tee}Number of unicast objectFIFO: {unicast_fifo_count}")
        print(f"{prt_last}Number of multicast objectFIFO: {multicast_fifo_count}")
        print(f"Number of objectFIFO link: {link_count}")

    result_file.write(f", {fifo_count}, {unicast_fifo_count}, {multicast_fifo_count}, {link_count}")


def parse_output_log_file(log_file_path, result_file, regex, prefix="", verbose=False):
    runtime = -1.0
    with open(log_file_path, 'r') as f:
        for line in f:
            match = re.search(regex, line)
            if match:
                runtime = float(match.group(1))
                break

    if verbose:
        print(f"{prefix}: {runtime:.2f} us")

    result_file.write(f", {runtime:.2f}")
    return runtime


def parse_routing_summary_json_file(json_file_path, result_file, verbose=False):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Count neighbor connections
    nbr_routes_count = 0
    if 'nbr_routes' in data:
        nbr_routes_count = len(data['nbr_routes'])

    # Count circuit switch connections
    cct_route_count = 0
    if 'cct_routes' in data:
        cct_route_count = len(data['cct_routes'])

    # Count packet flow connections
    pkt_route_count = 0
    if 'pkt_routes' in data:
        pkt_flow_count = len(data['pkt_routes'])
    
    # Count number of circuit switch track used
    total_cct_route_tracks = 0
    largest_cct_route_net = 0
    longest_cct_route_path = 0
    smallest_cct_route_net = sys.maxsize
    shortest_cct_route_path = sys.maxsize
    num_tracks_per_cct_route_net = []
    avg_num_tracks_per_cct_route_net = 0
    for cct_route in data.get('cct_routes', []):
        unique_tracks = set()   # account for track sharing in multicast
        for path in cct_route.get("intermediates", []):
            for node_idx, node in enumerate(path[:-1]):
                segment_id = (node["col_x"], node["row_y"], path[node_idx + 1]["col_x"], path[node_idx + 1]["row_y"])
                unique_tracks.add(segment_id)
        total_cct_route_tracks += len(unique_tracks)
        largest_cct_route_net = max(largest_cct_route_net, len(unique_tracks))
        longest_cct_route_path = max(longest_cct_route_path, len(path)-1)
        smallest_cct_route_net = min(smallest_cct_route_net, len(unique_tracks))
        shortest_cct_route_path = min(shortest_cct_route_path, len(path)-1)
        num_tracks_per_cct_route_net.append(len(unique_tracks))

    if num_tracks_per_cct_route_net:
        avg_num_tracks_per_cct_route_net = sum(num_tracks_per_cct_route_net) / len(num_tracks_per_cct_route_net)

    # Count number of packet flow track used
    total_pkt_route_tracks = 0
    largest_pkt_route_net = 0
    longest_pkt_route_path = 0
    smallest_pkt_route_net = sys.maxsize
    shortest_pkt_route_path = sys.maxsize
    num_tracks_per_pkt_route_net = []
    avg_num_tracks_per_pkt_route_net = 0
    for pkt_route in data.get('pkt_routes', []):
        unique_tracks = set()   # account for track sharing in multicast
        for path in pkt_route.get("intermediates", []):
            for node_idx, node in enumerate(path[:-1]):
                segment_id = (node["col_x"], node["row_y"], path[node_idx + 1]["col_x"], path[node_idx + 1]["row_y"])
                unique_tracks.add(segment_id)
        total_pkt_route_tracks += len(unique_tracks)
        largest_pkt_route_net = max(largest_pkt_route_net, len(unique_tracks))
        longest_pkt_route_path = max(longest_pkt_route_path, len(path)-1)
        smallest_pkt_route_net = min(smallest_pkt_route_net, len(unique_tracks))
        shortest_pkt_route_path = min(shortest_pkt_route_path, len(path)-1)
        num_tracks_per_pkt_route_net.append(len(unique_tracks))

    if num_tracks_per_pkt_route_net:
        avg_num_tracks_per_pkt_route_net = sum(num_tracks_per_pkt_route_net) / len(num_tracks_per_pkt_route_net)

    # Count number of DMA port Usage
    total_dma_ports = 0
    total_dma_in_ports = 0
    total_dma_out_ports = 0
    dma_ports = dict()
    for cct_route in data.get('cct_routes', []):
        x = cct_route["src"]["col_x"]
        y = cct_route["src"]["row_y"]
        total_dma_ports += 1
        total_dma_out_ports += 1
        if (x, y) in dma_ports:
            dma_ports[(x, y)]["out_count"] += 1
        else:
            dma_ports[(x, y)] = dict(in_count=0, out_count=1)

        for dst in cct_route["dsts"]:
            x = dst["col_x"]
            y = dst["row_y"]
            total_dma_ports += 1
            total_dma_in_ports += 1
            if (x, y) in dma_ports:
                dma_ports[(x, y)]["in_count"] += 1
            else:
                dma_ports[(x, y)] = dict(in_count=1, out_count=0)
            
    # Calculate total buffer size
    buffer_on_mem = []
    buffer_on_compute = []
    total_buffer_size = 0
    total_buffer_on_mem = 0
    total_buffer_on_compute = 0
    avg_buffer_size_on_mem = 0
    avg_buffer_size_on_compute = 0
    for buffer in data['buffers']:
        total_buffer_size += buffer.get('total_size_bytes', 0)
        if buffer.get('row_y') == 1:
            buffer_on_mem.append(buffer.get('total_size_bytes', 0))
        elif buffer.get('row_y') > 1:
            buffer_on_compute.append(buffer.get('total_size_bytes', 0))
    total_buffer_on_mem = sum(buffer_on_mem)
    total_buffer_on_compute = sum(buffer_on_compute)
    if buffer_on_mem:
        avg_buffer_size_on_mem = total_buffer_on_mem / len(buffer_on_mem)
    if buffer_on_compute:
        avg_buffer_size_on_compute = total_buffer_on_compute / len(buffer_on_compute)

    if verbose:
        print(f"Number of neighbour connections: {nbr_routes_count}")
        print(f"Number of circuit switch connections: {cct_route_count}")
        print(f"Number of packet flow connections: {pkt_route_count}")
        print(f"Total circuit switch tracks: {total_cct_route_tracks}")
        print(f"{prt_tee}Largest circuit switch net: {largest_cct_route_net}")
        print(f"{prt_tee}Smallest circuit switch net: {smallest_cct_route_net}")
        print(f"{prt_tee}Average number of track per net: {avg_num_tracks_per_cct_route_net:.2f}")
        print(f"{prt_tee}Longest circuit switch path: {longest_cct_route_path}")
        print(f"{prt_last}Shortest circuit switch path: {shortest_cct_route_path}")
        print(f"Total packet flow tracks: {total_pkt_route_tracks}")
        print(f"{prt_tee}Largest packet flow net: {largest_pkt_route_net}")
        print(f"{prt_tee}Smallest packet flow net: {smallest_pkt_route_net}")
        print(f"{prt_tee}Average number of track per net: {avg_num_tracks_per_pkt_route_net:.2f}")
        print(f"{prt_tee}Longest packet flow path: {longest_pkt_route_path}")
        print(f"{prt_last}Shortest packet flow path: {shortest_pkt_route_path}")
        print(f"Total DMA ports: {total_dma_ports}")
        print(f"{prt_tee}Total DMA in port used: {total_dma_in_ports}")
        print(f"{prt_last}Total DMA out port used: {total_dma_out_ports}")
        print(f"Total buffer size: {total_buffer_size} bytes")
        print(f"{prt_tee}Total buffer size on memory: {total_buffer_on_mem} bytes")
        print(f"{prt_tee}Average buffer size on memory: {avg_buffer_size_on_mem:.2f} bytes")
        print(f"{prt_tee}Total buffer size on compute: {total_buffer_on_compute} bytes")
        print(f"{prt_last}Average buffer size on compute: {avg_buffer_size_on_compute:.2f} bytes")

    # Append to results.csv
    result_file.write(f", {nbr_routes_count}, {cct_route_count}, {pkt_flow_count}")
    result_file.write(f", {total_cct_route_tracks}, {largest_cct_route_net}, {smallest_cct_route_net}, {avg_num_tracks_per_cct_route_net:.2f}, {longest_cct_route_path}, {shortest_cct_route_path}")
    result_file.write(f", {total_pkt_route_tracks}, {largest_pkt_route_net}, {smallest_pkt_route_net}, {avg_num_tracks_per_pkt_route_net:.2f}, {longest_pkt_route_path}, {shortest_pkt_route_path}")
    result_file.write(f", {total_dma_ports}, {total_dma_in_ports}, {total_dma_out_ports}")
    result_file.write(f", {total_buffer_size}, {total_buffer_on_mem}, {avg_buffer_size_on_mem:.2f}, {total_buffer_on_compute}, {avg_buffer_size_on_compute:.2f}")


def collect_results(output_dir = "build", result_file_path = 'results.csv', verbose=False):
    # Create the results file and write the header
    result_file = open(result_file_path, 'w')
    result_file.write("benchmark, task_name, num_objectFIFO, num_unicast_objectFIFO, num_multicast_objectFIFO, num_objectFIFO_link")
    result_file.write(", build_time [s], pnr_time [s], compilation_time [s], total_end2end_compilation_time [s], avg_runtime [us]")
    result_file.write(", num_neighbour_sharing_objectFIFO, num_circuit_switch_objectFIFO, num_packet_flow_objectFIFO")
    result_file.write(", total_num_circuit_switch_tracks, largest_circuit_switch_net, smallest_circuit_switch_net, avg_track_per_circuit_switch_net, longest_circuit_switch_path, shortest_circuit_switch_path")
    result_file.write(", total_num_packet_flow_tracks, largest_packet_flow_net, smallest_packet_flow_net, avg_track_per_packet_flow_net, longest_packet_flow_path, shortest_packet_flow_path")
    result_file.write(", total_num_dma_ports, total_num_dma_in_ports, total_num_dma_out_ports")
    result_file.write(", total_buffer_size [bytes], total_buffer_size_on_mem [bytes], avg_buffer_size_on_mem [bytes], total_buffer_size_on_compute [bytes], avg_buffer_size_on_compute [bytes]")
    result_file.write("\n")

    # Parse each benchmark directory
    benchmark_output_dir = glob.glob(output_dir + "/*")
    for benchmark_dir in benchmark_output_dir:
        if os.path.isdir(benchmark_dir):
            benchmark_name = os.path.basename(benchmark_dir)
            task_output_dir = glob.glob(benchmark_dir + "/*")
            task_names = [os.path.basename(task_path) for task_path in task_output_dir if os.path.isdir(task_path)]
            assert len(task_names) == len(task_output_dir), f"Non task_name directories exits in benchmark {benchmark_name}"
            # Sort task names in natural order (e.g., 1, 2, 10, 20)
            def natural_key(s):
                return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
            task_names.sort(key=natural_key)
            task_output_dir.sort(key=natural_key)

            # Parse each task within the task directory
            for task_output_path, task_name in zip(task_output_dir, task_names):
                mlir_file_path, build_log_file_path, pnr_log_file_path, aiecc_compile_log_file_path, npu_run_log_file_path, routing_summary_json_file_path = "", "", "", "", "", ""
                mlir_file_path = os.path.join(task_output_path, "build", f"{benchmark_name}.mlir")
                build_log_file_path = os.path.join(task_output_path, f"{task_name}.build.log")
                pnr_log_file_path = os.path.join(task_output_path, f"pnr.log")
                aiecc_compile_log_file_path = os.path.join(task_output_path, f"{task_name}.compile.log")
                npu_run_log_file_path = os.path.join(task_output_path, f"{task_name}.run.log")
                routing_summary_json_file_path = os.path.join(task_output_path, "build", f"post_compile_routing_summary.json")
                if not os.path.exists(npu_run_log_file_path):
                    npu_run_log_file_path = os.path.join(task_output_path, f"{task_name}.error.log")

                end2end_compilation_time = 0.0
                result_file.write(f"{benchmark_name}, {task_name}")
                # Parse MLIR file for number of objectFIFO and number of links info
                if os.path.exists(mlir_file_path):
                    parse_mlir_file(mlir_file_path, result_file, verbose)
                else:
                    result_file.write(f", N/A, N/A, N/A, N/A")
                # Parse build log for build time
                if os.path.exists(build_log_file_path):
                    build_time = parse_output_log_file(build_log_file_path, result_file, r'Build took[:\s]*([0-9]+(?:\.[0-9]+)?)\s*(?:secs?|seconds?)', "Build time", verbose)
                    if build_time >= 0.0:
                        end2end_compilation_time += build_time
                else:
                    result_file.write(f", N/A")
                # Parse PnR log for PnR time
                # Since PnR runtime is included in the build time, we do not add it to the end-to-end compilation time again
                if os.path.exists(pnr_log_file_path):
                    parse_output_log_file(pnr_log_file_path, result_file, r'Cost evaluation.*?:\s*([0-9]+(?:\.[0-9]+)?)\s*(?:secs?|seconds?)', "PnR time", verbose)
                else:
                    result_file.write(f", 0.0")
                # Parse AIECC compile log for compilation time
                if os.path.exists(aiecc_compile_log_file_path):
                    aiecc_compile_time = parse_output_log_file(aiecc_compile_log_file_path, result_file, r'Compile took[:\s]*([0-9]+(?:\.[0-9]+)?)\s*(?:secs?|seconds?)', "Compilation time", verbose)
                    if aiecc_compile_time >= 0.0:
                        end2end_compilation_time += aiecc_compile_time
                else:
                    result_file.write(f", N/A")
                # Write total end-to-end compilation time
                result_file.write(f", {end2end_compilation_time:.2f}")
                # Parse NPU run log for average runtime
                if os.path.exists(npu_run_log_file_path):
                    parse_output_log_file(npu_run_log_file_path, result_file, r'Avg NPU time:\s*([\d.]+)\s*us', "Avg NPU time", verbose)
                else:
                    result_file.write(f", N/A")
                # Parse routing summary JSON file for final placement and routing statistics
                if os.path.exists(routing_summary_json_file_path):
                    parse_routing_summary_json_file(routing_summary_json_file_path, result_file, verbose)
                else:
                    result_file.write(f", N/A, N/A, N/A")
                    result_file.write(f", N/A, N/A, N/A, N/A, N/A, N/A")
                    result_file.write(f", N/A, N/A, N/A, N/A, N/A, N/A")
                    result_file.write(f", N/A, N/A, N/A")
                    result_file.write(f", N/A, N/A, N/A, N/A, N/A")
                result_file.write(f"\n")

    result_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parse benchmark results and generate a summary CSV file.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="build",
        help="Directory containing benchmark output folders.",
        required=False,
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results.csv",
        help="Output CSV file path.",
        required=False,
    )
    args = parser.parse_args()
    
    collect_results(args.input_dir, args.output_csv)