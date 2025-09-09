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


def parse_output_log_file(output_log_file_path, result_file, verbose=False):
    avg_runtime = 0
    with open(output_log_file_path, 'r') as f:
        for line in f:
            avg_runtime_match = re.search(r'Avg NPU time:\s*([\d.]+)\s*us', line)
            if avg_runtime_match:
                avg_runtime = float(avg_runtime_match.group(1))
                break

    if verbose:
        print(f"Avg NPU runtime: {avg_runtime:.2f} us")

    result_file.write(f", {avg_runtime:.2f}")


def parse_routing_summary_json_file(json_file_path, result_file, verbose=False):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
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

    # Count neighbor connections
    nbr_routes_count = 0
    if 'nbr_routes' in data:
        nbr_routes_count = len(data['nbr_routes'])

    # Count circuit switch connections
    cct_routes_count = 0
    if 'cct_routes' in data:
        cct_routes_count = len(data['cct_routes'])

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

    # Count number of circuit switch track used
    total_circuit_switch_tracks = 0
    largest_net = 0
    longest_path = 0
    smallest_net = sys.maxsize
    shortest_path = sys.maxsize
    num_tracks_per_net = []
    avg_num_tracks_per_net = 0
    for cct_route in data.get('cct_routes', []):
        unique_tracks = set()   # account for track sharing in multicast
        for path in cct_route.get("intermediates", []):
            for node_idx, node in enumerate(path[:-1]):
                segment_id = (node["col_x"], node["row_y"], path[node_idx + 1]["col_x"], path[node_idx + 1]["row_y"])
                unique_tracks.add(segment_id)
        total_circuit_switch_tracks += len(unique_tracks)
        largest_net = max(largest_net, len(unique_tracks))
        longest_path = max(longest_path, len(path)-1)
        smallest_net = min(smallest_net, len(unique_tracks))
        shortest_path = min(shortest_path, len(path)-1)
        num_tracks_per_net.append(len(unique_tracks))

    if num_tracks_per_net:
        avg_num_tracks_per_net = sum(num_tracks_per_net) / len(num_tracks_per_net)

    if verbose:
        print(f"Number of neighbour connections: {nbr_routes_count}")
        print(f"Number of circuit switch connections: {cct_routes_count}")
        print(f"Total circuit switch tracks: {total_circuit_switch_tracks}")
        print(f"{prt_tee}Largest circuit switch net: {largest_net}")
        print(f"{prt_tee}Smallest circuit switch net: {smallest_net}")
        print(f"{prt_tee}Average number of track per net: {avg_num_tracks_per_net:.2f}")
        print(f"{prt_tee}Longest circuit switch path: {longest_path}")
        print(f"{prt_last}Shortest circuit switch path: {shortest_path}")
        print(f"Total DMA ports: {total_dma_ports}")
        print(f"{prt_tee}Total DMA in port used: {total_dma_in_ports}")
        print(f"{prt_last}Total DMA out port used: {total_dma_out_ports}")
        print(f"Total buffer size: {total_buffer_size} bytes")
        print(f"{prt_tee}Total buffer size on memory: {total_buffer_on_mem} bytes")
        print(f"{prt_tee}Average buffer size on memory: {avg_buffer_size_on_mem:.2f} bytes")
        print(f"{prt_tee}Total buffer size on compute: {total_buffer_on_compute} bytes")
        print(f"{prt_last}Average buffer size on compute: {avg_buffer_size_on_compute:.2f} bytes")

    # Append to results.csv
    result_file.write(f", {nbr_routes_count}, {cct_routes_count}")
    result_file.write(f", {total_circuit_switch_tracks}, {largest_net}, {smallest_net}, {avg_num_tracks_per_net:.2f}, {longest_path}, {shortest_path}")
    result_file.write(f", {total_dma_ports}, {total_dma_in_ports}, {total_dma_out_ports}")
    result_file.write(f", {total_buffer_size}, {total_buffer_on_mem}, {avg_buffer_size_on_mem:.2f}, {total_buffer_on_compute}, {avg_buffer_size_on_compute:.2f}")


def collect_results(output_dir = "build", result_file_path = 'results.csv', variant = 'std', verbose=False):
    # Create the results file and write the header
    result_file = open(result_file_path, 'w')
    result_file.write("benchmark, task_name, num_objectFIFO, num_unicast_objectFIFO, num_multicast_objectFIFO, num_objectFIFO_link")
    result_file.write(", avg_runtime [us]")
    result_file.write(", num_neighbour_sharing_objectFIFO, num_circuit_switch_objectFIFO")
    result_file.write(", total_num_circuit_switch_tracks, largest_circuit_switch_net, smallest_circuit_switch_net, avg_track_per_circuit_switch_net, longest_circuit_switch_path, shortest_circuit_switch_path")
    result_file.write(", total_num_dma_ports, total_num_dma_in_ports, total_num_dma_out_ports")
    result_file.write(", total_buffer_size [bytes], total_buffer_size_on_mem [bytes], avg_buffer_size_on_mem [bytes], total_buffer_size_on_compute [bytes], avg_buffer_size_on_compute [bytes]")
    result_file.write("\n")

    # Parse each benchmark directory
    benchmark_output_dir = glob.glob(output_dir + "/*")
    for benchmark_dir in benchmark_output_dir:
        if os.path.isdir(benchmark_dir):
            benchmark_name = os.path.basename(benchmark_dir)
            task_files = glob.glob(benchmark_dir + "/*.build.mlir")
            task_names = [os.path.basename(f).split('.')[0] for f in task_files]
            task_names.sort()

            # Parse each task within the benchmark directory
            for task_name in task_names:
                mlir_file_path, output_log_file_path, json_file_path = "", "", ""
                if variant == 'std':
                    mlir_file_path = os.path.join(benchmark_dir, f"{task_name}.build.mlir")
                    output_log_file_path = os.path.join(benchmark_dir, f"{task_name}.stdout.run.log")
                    json_file_path = os.path.join(benchmark_dir, f"{task_name}.route_summary.build.json")
                elif variant == 'pnr':
                    mlir_file_path = os.path.join(benchmark_dir, f"{task_name}.pnr.mlir")
                    output_log_file_path = os.path.join(benchmark_dir, f"{task_name}.stdout.run.log")
                    json_file_path = os.path.join(benchmark_dir, f"{task_name}.route_summary.pnr.json")

                result_file.write(f"{benchmark_name}, {task_name}")
                if os.path.exists(mlir_file_path):
                    parse_mlir_file(mlir_file_path, result_file, verbose)
                else:
                    result_file.write(f", N/A, N/A, N/A, N/A")
                if os.path.exists(output_log_file_path):
                    parse_output_log_file(output_log_file_path, result_file, verbose)
                else:
                    result_file.write(f", N/A")
                if os.path.exists(json_file_path):
                    parse_routing_summary_json_file(json_file_path, result_file, verbose)
                else:
                    result_file.write(f", N/A, N/A")
                    result_file.write(f", N/A, N/A, N/A, N/A, N/A, N/A")
                    result_file.write(f", N/A, N/A, N/A")
                    result_file.write(f", N/A, N/A, N/A, N/A, N/A")
                result_file.write(f"\n")

    result_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parse benchmark results and generate a summary CSV file.")
    parser.add_argument(
        "--variant",
        type=str,
        default="std",
        help="Specify which variant to parse: 'std' for standard flow, 'pnr' for place-and-route results.",
        required=False,
    )
    args = parser.parse_args()

    collect_results("build", f"{args.variant}_results.csv", args.variant)
