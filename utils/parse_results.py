import sys
import re
import json
import os
import glob

# prefix components:
prt_space =  '    '
prt_branch = '│   '
# pointers:
prt_tee =    '├── '
prt_last =   '└── '

def parse_mlir_file(filename):
    fifo_count = 0
    unicast_fifo_count = 0
    multicast_fifo_count = 0
    link_count = 0

    with open(filename, 'r') as f:
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

    print(f"Number of objectFIFO: {fifo_count}")
    print(f"{prt_tee}Number of unicast objectFIFO: {unicast_fifo_count}")
    print(f"{prt_last}Number of multicast objectFIFO: {multicast_fifo_count}")
    print(f"Number of objectFIFO link: {link_count}")

    with open('results.csv', 'w') as out_file:
        out_file.write(f"3D Mesh, {fifo_count}, {unicast_fifo_count}, {multicast_fifo_count}, {link_count}")


def parse_routing_summary_json_file(filename):
    with open(filename, 'r') as f:
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
    with open('results.csv', 'a') as out_file:
        out_file.write(f", {nbr_routes_count}, {cct_routes_count}")
        out_file.write(f", {total_circuit_switch_tracks}, {largest_net}, {smallest_net}, {avg_num_tracks_per_net:.2f}, {longest_path}, {shortest_path}")
        out_file.write(f", {total_dma_ports}, {total_dma_in_ports}, {total_dma_out_ports}")
        out_file.write(f", {total_buffer_size}, {total_buffer_on_mem}, {avg_buffer_size_on_mem:.2f}, {total_buffer_on_compute}, {avg_buffer_size_on_compute:.2f}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Look for MLIR file in the build directory
        build_dir = os.path.join(os.getcwd(), 'build')
        mlir_files = glob.glob(os.path.join(build_dir, '*.mlir'))
        mlir_file = mlir_files[0] if mlir_files else ''
        if os.path.exists(mlir_file):
            parse_mlir_file(mlir_file)
        else:
            print("Usage: python parse_results.py <file.mlir> <file.json>")
            print("No MLIR file found. Provide as first argument or place .mlir file in build directory.")
    else:
        if os.path.exists(sys.argv[1]):
            parse_mlir_file(sys.argv[1])
        else:
            print("Usage: python parse_results.py <file.mlir> <file.json>")
            print("No MLIR file found. Provide as first argument or place .mlir file in build directory.")

    # Parse JSON file if provided
    if len(sys.argv) < 3:
        # Look for route_summary.json in the build directory
        build_dir = os.path.join(os.getcwd(), 'build')
        json_file = os.path.join(build_dir, 'route_summary.json')
        if os.path.exists(json_file):
            parse_routing_summary_json_file(json_file)
        else:
            print("Usage: python parse_results.py <file.mlir> <file.json>")
            print("No JSON file found. Provide as second argument or place route_summary.json in build directory.")
    else:
        if os.path.exists(sys.argv[2]):
            parse_routing_summary_json_file(sys.argv[2])
        else:
            print("Usage: python parse_results.py <file.mlir> <file.json>")
            print("No JSON file found. Provide as second argument or place route_summary.json in build directory.")