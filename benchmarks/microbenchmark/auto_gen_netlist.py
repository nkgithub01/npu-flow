import json
import argparse

def write_netlist_to_file(netlist, output_file):
    with open(output_file, 'w') as netlist_file:
        json.dump(netlist, netlist_file, indent=4)


def helper_connect_nodes(netlist, net_id, src_node_id, dst_node_ids, args, need_linking=False, link_src_net_id=0):
    net = {
        "net_id": net_id,
        "need_linking": need_linking,
        "link_src_net_id": link_src_net_id,
        "src_tile_id": src_node_id,
        "dst_tile_ids": dst_node_ids,
        "depths": [args.obj_fifo_depth],
        "byte_size_per_depth": args.obj_fifo_byte_size_per_depth,
    }
    netlist["nets"].append(net)


def generate_mesh_netlist(args):
    netlist = dict(
        nodes = [],
        nets = []
    )

    # Define nodes
    nodes_loc2ID_lookup = dict()
    for x in range(args.num_cols):
        for y in range(args.num_rows):
            if y == 0:
                type_str = "SHIM"
            elif y == 1:
                type_str = "MEM"
            else:
                type_str = "COMP"

            nodes_loc2ID_lookup[(x, y)] = len(netlist["nodes"])
            node = {
                "tile_id": len(netlist["nodes"]),
                "type": type_str,
                "col_x": x,
                "row_y": y,
            }
            netlist["nodes"].append(node)

    # Define nets connections
    # Each node is connected to its right and top neighbor
    for x in range(args.num_cols):
        for y in range(args.num_rows):
            node_id = nodes_loc2ID_lookup[(x, y)]
            # The SHIM node is connected to the MEM node in the same column and linked to the net from MEM to the compute core above it
            if y == 1:
                # Create a net for the MEM node in the same column
                same_col_SHIM_id = nodes_loc2ID_lookup[(x, 0)]
                same_col_MEM_id = nodes_loc2ID_lookup[(x, 1)]
                same_col_COMP_id = nodes_loc2ID_lookup[(x, 2)]
                helper_connect_nodes(netlist, len(netlist["nets"]), same_col_SHIM_id, [same_col_MEM_id], args)
                # Link this net to the MEM node in the same column
                helper_connect_nodes(netlist, len(netlist["nets"]), same_col_MEM_id, [same_col_COMP_id], args, need_linking=True, link_src_net_id=len(netlist["nets"]) - 1)

            # The middle region compute core is connected to its right and top neighbors
            if x < args.num_cols - 1 and 1 < y < args.num_rows - 1:
                # Create a net for the right neighbor
                right_neighbor_id = nodes_loc2ID_lookup[(x + 1, y)]
                helper_connect_nodes(netlist, len(netlist["nets"]), node_id, [right_neighbor_id], args)

                # Create a net for the top neighbor
                top_neighbor_id = nodes_loc2ID_lookup[(x, y + 1)]
                helper_connect_nodes(netlist, len(netlist["nets"]), node_id, [top_neighbor_id], args)
            
            # The rightmost column nodes are connected to their top neighbor
            if x == args.num_cols - 1 and 1 < y < args.num_rows - 1:
                # Create a net for the top neighbor
                top_neighbor_id = nodes_loc2ID_lookup[(x, y + 1)]
                helper_connect_nodes(netlist, len(netlist["nets"]), node_id, [top_neighbor_id], args)
            
            # The top row nodes are connected to their right neighbor
            if y == args.num_rows - 1 and x < args.num_cols - 1:
                # Create a net for the right neighbor
                right_neighbor_id = nodes_loc2ID_lookup[(x + 1, y)]
                helper_connect_nodes(netlist, len(netlist["nets"]), node_id, [right_neighbor_id], args)
            
            # The top row nodes are also connected back to MEM node in the same column
            if y == args.num_rows - 1:
                # Create a net for the MEM node in the same column
                same_col_MEM_id = nodes_loc2ID_lookup[(x, 1)]
                helper_connect_nodes(netlist, len(netlist["nets"]), node_id, [same_col_MEM_id], args)
                # Link this net to the SHIM node in the same column
                same_col_SHIM_id = nodes_loc2ID_lookup[(x, 0)]
                helper_connect_nodes(netlist, len(netlist["nets"]), same_col_MEM_id, [same_col_SHIM_id], args, need_linking=True, link_src_net_id=len(netlist["nets"]) - 1)
    
    return netlist


def generate_netlist(args):
    try:
        generate_func = TOPOLOGIES_CONVERSION[args.netlist_topologies]
        netlist = generate_func(args)
    except KeyError:
        raise ValueError(f"Unsupported netlist topology: {args.netlist_topologies}. Supported topologies: {list(TOPOLOGIES_CONVERSION.keys())}")
    
    return netlist


def main(args):

    # Generate netlist for AIE data flow
    netlist = generate_netlist(args)

    # Write the output netlist
    write_netlist_to_file(netlist, args.output_netlist)


TOPOLOGIES_CONVERSION = {
    "mesh": generate_mesh_netlist,
}


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        prog="auto_gen_netlist",
        description="Script to auto-generate simple netlist topologies for AIE data flow",
    )
    argparser.add_argument(
        "-nl",
        "--netlist", 
        type=str, 
        dest="output_netlist",
        default="AIE_data_flow_netlists/netlist.json",
    )
    argparser.add_argument(
        "--topology", 
        type=str, 
        dest="netlist_topologies",
        choices=list(TOPOLOGIES_CONVERSION.keys()),
        default="mesh",
    )
    argparser.add_argument(
        "--dtype", 
        type=str, 
        dest="dtype_str",
        choices=["i8", "i16", "i32"], 
        default="i32"
    )
    argparser.add_argument(
        "--obj_fifo_depth", 
        type=int, 
        dest="obj_fifo_depth",
        default=5
    )
    argparser.add_argument(
        "--obj_fifo_byte_size_per_depth", 
        type=int, 
        dest="obj_fifo_byte_size_per_depth",
        default=4
    )
    argparser.add_argument(
        "--num_rows", 
        type=int, 
        dest="num_rows",
        default=6
    )
    argparser.add_argument(
        "--num_cols", 
        type=int, 
        dest="num_cols",
        default=8
    )

    opts = argparser.parse_args()
    main(opts)
