import json
import argparse

def write_netlist_to_file(netlist, output_file):
    with open(output_file, 'w') as netlist_file:
        json.dump(netlist, netlist_file, indent=4)


def helper_connect_nodes(netlist, net_id, src_node_id, dst_node_ids, args, need_linking=False, link_src_net_id=None):
    if link_src_net_id is None:
        link_src_net_id = [0]
    if not isinstance(link_src_net_id, list):
        link_src_net_id = [link_src_net_id]

    net = {
        "net_id": net_id,
        "need_linking": need_linking,
        "link_src_net_ids": link_src_net_id,
        "src_tile_id": src_node_id,
        "dst_tile_ids": dst_node_ids,
        "depths": [args.obj_fifo_depth],
        "byte_size_per_depth": args.obj_fifo_byte_size_per_depth,
    }
    netlist["nets"].append(net)


def generate_mesh_topology_netlist(args):
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
    # Each compute node is connected to its right and top neighbor
    for x in range(args.num_cols):
        for y in range(args.num_rows):
            node_id = nodes_loc2ID_lookup[(x, y)]
            # The SHIM node is connected to the MEM node in the same column and linked to the net from MEM to the compute core above it
            if y == 1:
                # Create a net to the MEM node in the same column
                same_col_SHIM_id = nodes_loc2ID_lookup[(x, 0)]
                same_col_MEM_id = nodes_loc2ID_lookup[(x, 1)]
                same_col_COMP_id = nodes_loc2ID_lookup[(x, 2)]
                helper_connect_nodes(netlist, len(netlist["nets"]), same_col_SHIM_id, [same_col_MEM_id], args)
                # Link this net to the MEM node in the same column
                helper_connect_nodes(netlist, len(netlist["nets"]), same_col_MEM_id, [same_col_COMP_id], args, need_linking=True, link_src_net_id=len(netlist["nets"]) - 1)

            # The middle region compute core is connected to its right and top neighbors
            if x < args.num_cols - 1 and 1 < y < args.num_rows - 1:
                # Create a net to the right neighbor
                right_neighbor_id = nodes_loc2ID_lookup[(x + 1, y)]
                helper_connect_nodes(netlist, len(netlist["nets"]), node_id, [right_neighbor_id], args)

                # Create a net to the top neighbor
                top_neighbor_id = nodes_loc2ID_lookup[(x, y + 1)]
                helper_connect_nodes(netlist, len(netlist["nets"]), node_id, [top_neighbor_id], args)
            
            # The rightmost column nodes are connected to their top neighbor
            if x == args.num_cols - 1 and 1 < y < args.num_rows - 1:
                # Create a net to the top neighbor
                top_neighbor_id = nodes_loc2ID_lookup[(x, y + 1)]
                helper_connect_nodes(netlist, len(netlist["nets"]), node_id, [top_neighbor_id], args)
            
            # The top row nodes are connected to their right neighbor
            if y == args.num_rows - 1 and x < args.num_cols - 1:
                # Create a net to the right neighbor
                right_neighbor_id = nodes_loc2ID_lookup[(x + 1, y)]
                helper_connect_nodes(netlist, len(netlist["nets"]), node_id, [right_neighbor_id], args)
            
            # The top row nodes are also connected back to MEM node in the same column
            if y == args.num_rows - 1:
                # Create a net to the MEM node in the same column
                same_col_MEM_id = nodes_loc2ID_lookup[(x, 1)]
                helper_connect_nodes(netlist, len(netlist["nets"]), node_id, [same_col_MEM_id], args)
                # Link this net to the SHIM node in the same column
                same_col_SHIM_id = nodes_loc2ID_lookup[(x, 0)]
                helper_connect_nodes(netlist, len(netlist["nets"]), same_col_MEM_id, [same_col_SHIM_id], args, need_linking=True, link_src_net_id=len(netlist["nets"]) - 1)
    
    return netlist


def generate_vertical_line_topology_netlist(args):
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
    # Each compute node is connected to its top neighbor
    for x in range(args.num_cols):
        for y in range(args.num_rows):
            node_id = nodes_loc2ID_lookup[(x, y)]
            # The SHIM node is connected to the MEM node in the same column and linked to the net from MEM to the compute core above it
            if y == 1:
                # Create a net to the MEM node in the same column
                same_col_SHIM_id = nodes_loc2ID_lookup[(x, 0)]
                same_col_MEM_id = nodes_loc2ID_lookup[(x, 1)]
                same_col_COMP_id = nodes_loc2ID_lookup[(x, 2)]
                helper_connect_nodes(netlist, len(netlist["nets"]), same_col_SHIM_id, [same_col_MEM_id], args)
                # Link this net to the MEM node in the same column
                helper_connect_nodes(netlist, len(netlist["nets"]), same_col_MEM_id, [same_col_COMP_id], args, need_linking=True, link_src_net_id=len(netlist["nets"]) - 1)

            # The middle region compute core is connected to its top neighbors
            elif 1 < y < args.num_rows - 1:
                # Create a net to the top neighbor
                top_neighbor_id = nodes_loc2ID_lookup[(x, y + 1)]
                helper_connect_nodes(netlist, len(netlist["nets"]), node_id, [top_neighbor_id], args)
            
            # The top row nodes are also connected back to MEM node in the same column
            elif y == args.num_rows - 1:
                # Create a net to the MEM node in the same column
                same_col_MEM_id = nodes_loc2ID_lookup[(x, 1)]
                helper_connect_nodes(netlist, len(netlist["nets"]), node_id, [same_col_MEM_id], args)
                # Link this net to the SHIM node in the same column
                same_col_SHIM_id = nodes_loc2ID_lookup[(x, 0)]
                helper_connect_nodes(netlist, len(netlist["nets"]), same_col_MEM_id, [same_col_SHIM_id], args, need_linking=True, link_src_net_id=len(netlist["nets"]) - 1)
    
    return netlist


def generate_horizontal_line_topology_netlist(args):
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
    # Each compute node is connected to its right neighbor
    min_dim = min(args.num_cols, args.num_rows)
    for x in range(args.num_cols):
        for y in range(2, args.num_rows):
            node_id = nodes_loc2ID_lookup[(x, y)]
            # The SHIM node is connected to the MEM node in the same column and linked to the net from MEM to the compute core in the left column
            if x == 0:
                # Create a net to the MEM node in the same column
                COMP_y_map2_SHIM_x = (y - 2) % min_dim
                COMP_y_map2_MEM_x = COMP_y_map2_SHIM_x
                SHIM_id = nodes_loc2ID_lookup[(COMP_y_map2_SHIM_x, 0)]
                MEM_id = nodes_loc2ID_lookup[(COMP_y_map2_MEM_x, 1)]
                COMP_id = nodes_loc2ID_lookup[(x, y)]
                helper_connect_nodes(netlist, len(netlist["nets"]), SHIM_id, [MEM_id], args)
                # Link this net to the MEM node in the same column
                helper_connect_nodes(netlist, len(netlist["nets"]), MEM_id, [COMP_id], args, need_linking=True, link_src_net_id=len(netlist["nets"]) - 1)

            # The middle region compute core is connected to its right neighbors
            if x < args.num_cols - 1:
                # Create a net to the right neighbor
                right_neighbor_id = nodes_loc2ID_lookup[(x + 1, y)]
                helper_connect_nodes(netlist, len(netlist["nets"]), node_id, [right_neighbor_id], args)
            
            # The rightmost column nodes are also connected back to MEM node
            elif x == args.num_cols - 1:
                # Create a net to the MEM node
                COMP_y_map2_SHIM_x = (args.num_cols - 1) - (y - 2) % min_dim
                COMP_y_map2_MEM_x = COMP_y_map2_SHIM_x
                SHIM_id = nodes_loc2ID_lookup[(COMP_y_map2_SHIM_x, 0)]
                MEM_id = nodes_loc2ID_lookup[(COMP_y_map2_MEM_x, 1)]
                COMP_id = nodes_loc2ID_lookup[(x, y)]
                helper_connect_nodes(netlist, len(netlist["nets"]), COMP_id, [MEM_id], args)
                # Link this net to the SHIM node in the same column
                helper_connect_nodes(netlist, len(netlist["nets"]), MEM_id, [SHIM_id], args, need_linking=True, link_src_net_id=len(netlist["nets"]) - 1)
    
    return netlist


def generate_tree_topology_netlist(args):
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
    # Each compute node is connected to 2 other nodes to form a binary tree structure
    num_node = args.num_cols * (args.num_rows - 2)
    for idx in range(num_node):
        x = idx % args.num_cols
        y = 2 + (idx // args.num_cols) % (args.num_rows - 2)
        node_id = nodes_loc2ID_lookup[(x, y)]
        
        if idx == 0:
            # The root node get data from the SHIM node in the same column through the MEM node
            same_col_SHIM_id = nodes_loc2ID_lookup[(x, 0)]
            same_col_MEM_id = nodes_loc2ID_lookup[(x, 1)]
            helper_connect_nodes(netlist, len(netlist["nets"]), same_col_SHIM_id, [same_col_MEM_id], args)
            # Link this net to the SHIM node in the same column
            helper_connect_nodes(netlist, len(netlist["nets"]), same_col_MEM_id, [node_id], args, need_linking=True, link_src_net_id=len(netlist["nets"]) - 1)

        has_no_children = True
        # Connect to left child
        left_child_idx = idx * 2 + 1
        left_child_x = left_child_idx % args.num_cols
        left_child_y = 2 + (left_child_idx // args.num_cols)
        if left_child_x < args.num_cols and left_child_y < args.num_rows:
            left_child_id = nodes_loc2ID_lookup[(left_child_x, left_child_y)]
            helper_connect_nodes(netlist, len(netlist["nets"]), node_id, [left_child_id], args)
            has_no_children = False
        
        # Connect to right child
        right_child_idx = idx * 2 + 2
        right_child_x = right_child_idx % args.num_cols
        right_child_y = 2 + (right_child_idx // args.num_cols)
        if right_child_x < args.num_cols and right_child_y < args.num_rows:
            right_child_id = nodes_loc2ID_lookup[(right_child_x, right_child_y)]
            helper_connect_nodes(netlist, len(netlist["nets"]), node_id, [right_child_id], args)
            has_no_children = False
        
        if has_no_children:
            # If the node has no children, connect it to the MEM node in the same column
            same_col_MEM_id = nodes_loc2ID_lookup[(x, 1)]
            helper_connect_nodes(netlist, len(netlist["nets"]), node_id, [same_col_MEM_id], args)
            # Link this net to the SHIM node in the same column
            same_col_SHIM_id = nodes_loc2ID_lookup[(x, 0)]
            helper_connect_nodes(netlist, len(netlist["nets"]), same_col_MEM_id, [same_col_SHIM_id], args, need_linking=True, link_src_net_id=len(netlist["nets"]) - 1)

    return netlist


def generate_cnn_topology_netlist(args):
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
    # Input X is connected from SHIM node 0, 1, 2, 3 to MEM node 0, 1, 2, 3 to COMP nodes in the first column
    for idx in range(4):
        # Connect SHIM node to MEM node in the same column
        x = idx
        SHIM_id = nodes_loc2ID_lookup[(x, 0)]
        MEM_id = nodes_loc2ID_lookup[(x, 1)]
        helper_connect_nodes(netlist, len(netlist["nets"]), SHIM_id, [MEM_id], args)
        
        # Link this net to the Comp node in the first column
        y = 5 - idx
        COMP_id = nodes_loc2ID_lookup[(0, y)]
        helper_connect_nodes(netlist, len(netlist["nets"]), MEM_id, [COMP_id], args, need_linking=True, link_src_net_id=len(netlist["nets"]) - 1)

    # Input W is connected from SHIM node 0 to MEM node 0 to COMP nodes in the first row
    for idx in range(1):
        # Connect SHIM node to MEM node in the same column
        x = idx
        SHIM_id = nodes_loc2ID_lookup[(x, 0)]
        MEM_id = nodes_loc2ID_lookup[(x, 1)]
        helper_connect_nodes(netlist, len(netlist["nets"]), SHIM_id, [MEM_id], args)
        
        # Link this net to the Comp node in the first column
        COMP_ids = [nodes_loc2ID_lookup[(0, y)] for y in range(2, 6)]
        helper_connect_nodes(netlist, len(netlist["nets"]), MEM_id, COMP_ids, args, need_linking=True, link_src_net_id=len(netlist["nets"]) - 1)

    # The first column compute nodes are connected to the second column compute nodes
    for y in range(2, 6):
        # Get the node ID for the compute nodes
        first_col_COMP_id = nodes_loc2ID_lookup[(0, y)]
        second_col_COMP_id = nodes_loc2ID_lookup[(1, y)]
        
        # Create a net to connect the two compute nodes
        helper_connect_nodes(netlist, len(netlist["nets"]), first_col_COMP_id, [second_col_COMP_id], args)

    # The top 2 rows of compute nodes in the second column are connected to the top node in the thrid column
    # and the bottom 2 rows of compute nodes in the second column are connected to the bottom thrid from top compute node in the third column
    for y in range(2, 6):
        # Get the node ID for the compute nodes in the second column
        second_col_COMP_id = nodes_loc2ID_lookup[(1, y)]
        
        if y < 4:
            # Connect to the top third node in the third column
            third_col_COMP_id = nodes_loc2ID_lookup[(2, 3)]
        else:
            # Connect to the top node in the third column
            third_col_COMP_id = nodes_loc2ID_lookup[(2, 5)]
        helper_connect_nodes(netlist, len(netlist["nets"]), second_col_COMP_id, [third_col_COMP_id], args)

    # The third column compute nodes are connected in a way that row_5 to row_4, row_3 to row_2. row_4 and row_2 are connected to the MEM node in the same column
    # The connection to the MEM is forwarded to the COMP node in column 3, 4, 5, 6
    linked_src_net_ids = []
    for y in range(2, 6):
        # Get the node ID for the compute nodes in the third column
        third_col_COMP_id = nodes_loc2ID_lookup[(2, y)]
        
        if y == 5 or y == 3:
            # Connect to the Comp node below in the same column
            below_COMP_id = nodes_loc2ID_lookup[(2, y-1)]
            helper_connect_nodes(netlist, len(netlist["nets"]), third_col_COMP_id, [below_COMP_id], args)
        
        elif y == 4 or y == 2:
            # Connect to the MEM node in the same column
            same_col_MEM_id = nodes_loc2ID_lookup[(2, 1)]
            linked_src_net_ids.append(len(netlist["nets"]))
            helper_connect_nodes(netlist, len(netlist["nets"]), third_col_COMP_id, [same_col_MEM_id], args)
    # Link this net to the net to the COMP nodes in columns 3, 4, 5, 6
    for idx in range(1):
        # Get the node ID for the compute nodes in columns 3, 4, 5, 6
        COMP_ids = [nodes_loc2ID_lookup[(x, y)] for x in range(3, 7) for y in range(2, 6)]
        # Connect the net from MEM nodes in column 1 to the COMP nodes in columns 3, 4, 5, 6
        MEM_id = nodes_loc2ID_lookup[(2, 1)]
        helper_connect_nodes(netlist, len(netlist["nets"]), MEM_id, COMP_ids, args, need_linking=True, link_src_net_id=linked_src_net_ids)

    # The weight for Fully connected layers is sent in from SHIM nodes 2, 3, 4, 5 to MEM nodes 2, 3, 4, 5 to COMP nodes in columns 3, 4, 5, 6
    for x in range(2, 6):
        # Connect SHIM node to MEM node in the same column
        SHIM_id = nodes_loc2ID_lookup[(x, 0)]
        MEM_id = nodes_loc2ID_lookup[(x, 1)]
        helper_connect_nodes(netlist, len(netlist["nets"]), SHIM_id, [MEM_id], args)
        
        # Link this net to the Comp nodes in columns 3, 4, 5, 6
        COMP_ids = [nodes_loc2ID_lookup[(x+1, y)] for y in range(2, 6)]
        helper_connect_nodes(netlist, len(netlist["nets"]), MEM_id, COMP_ids, args, need_linking=True, link_src_net_id=len(netlist["nets"]) - 1)
    
    # The output from the compute nodes in columns 3, 4, 5, 6 are chained sequentially to the MEM node in column 7
    linked_src_net_ids = []
    for x in range(3, 7):
        for y in range(2, 6):
            # Get the node ID for the compute nodes in columns 3, 4, 5, 6
            COMP_id = nodes_loc2ID_lookup[(x, y)]

            if y == 2 and x == 6:
                # Connect to the MEM node
                MEM_id = nodes_loc2ID_lookup[(x, 1)]
                linked_src_net_ids.append(len(netlist["nets"]))
                helper_connect_nodes(netlist, len(netlist["nets"]), COMP_id, [MEM_id], args)
            elif y == 2:
                # Connect to the COMP node to the right in the same row
                right_COMP_id = nodes_loc2ID_lookup[(x+1, y)]
                helper_connect_nodes(netlist, len(netlist["nets"]), COMP_id, [right_COMP_id], args)
            else:
                # Connect to the COMP node below in the same column
                below_COMP_id = nodes_loc2ID_lookup[(x, y-1)]
                helper_connect_nodes(netlist, len(netlist["nets"]), COMP_id, [below_COMP_id], args)

    # Link this net to the net that sends data from the MEM node in column 6 to the COMP node in column 7
    for idx in range(1):
        # Get the node ID for the compute nodes in column 7
        COMP_ids = [nodes_loc2ID_lookup[(7, y)] for y in range(2, 6)]
        # Connect the net from MEM node in column 6 to the COMP node in column 7
        MEM_id = nodes_loc2ID_lookup[(6, 1)]
        helper_connect_nodes(netlist, len(netlist["nets"]), MEM_id, COMP_ids, args, need_linking=True, link_src_net_id=linked_src_net_ids)
    
    # The weight for the final Fully connected layer is sent in from SHIM node 6 to MEM node 6 to COMP nodes in column 7
    for idx in range(1):
        # Connect SHIM node to MEM node in the same column
        SHIM_id = nodes_loc2ID_lookup[(6, 0)]
        MEM_id = nodes_loc2ID_lookup[(6, 1)]
        helper_connect_nodes(netlist, len(netlist["nets"]), SHIM_id, [MEM_id], args)
        
        # Link this net to the Comp nodes in column 7
        COMP_ids = [nodes_loc2ID_lookup[(7, y)] for y in range(2, 6)]
        helper_connect_nodes(netlist, len(netlist["nets"]), MEM_id, COMP_ids, args, need_linking=True, link_src_net_id=len(netlist["nets"]) - 1)
    
    # The output of the COMP nodes in column 7 are sent to the COMP node below it. The output of COMP node at row 2 is sent to the MEM node then SHIM node in column 7
    for y in range(2, 5):
        # the output of the top 3 COMP nodes in column 7 are sent to the COMP node below it
        if y > 2:
            # Get the node ID for the compute nodes in column 7
            COMP_id1 = nodes_loc2ID_lookup[(7, y)]
            # Connect to the COMP node at row 2 in the same column
            COMP_id2 = nodes_loc2ID_lookup[(7, y-1)]
            helper_connect_nodes(netlist, len(netlist["nets"]), COMP_id1, [COMP_id2], args)
        
        # the output of COMP node at row 2 is sent to the MEM node then SHIM node in column 7
        elif y == 2:
            # Get the node ID for the compute nodes in column 7
            COMP_id = nodes_loc2ID_lookup[(7, y)]
            # Connect to the MEM node in the same column
            MEM_id = nodes_loc2ID_lookup[(7, 1)]
            helper_connect_nodes(netlist, len(netlist["nets"]), COMP_id, [MEM_id], args)
            # Link this net to the SHIM node in the same column
            SHIM_id = nodes_loc2ID_lookup[(7, 0)]
            helper_connect_nodes(netlist, len(netlist["nets"]), MEM_id, [SHIM_id], args, need_linking=True, link_src_net_id=len(netlist["nets"]) - 1)

    return netlist


def generate_netlist(args):
    if args.netlist_topologies not in TOPOLOGIES_CONVERSION:
        raise ValueError(f"Unsupported netlist topology: {args.netlist_topologies}. Supported topologies: {list(TOPOLOGIES_CONVERSION.keys())}")
    generate_func = TOPOLOGIES_CONVERSION[args.netlist_topologies]
    netlist = generate_func(args)
    
    return netlist


def main(args):

    # Generate netlist for AIE data flow
    netlist = generate_netlist(args)

    # Write the output netlist
    write_netlist_to_file(netlist, args.output_netlist)


TOPOLOGIES_CONVERSION = {
    "mesh": generate_mesh_topology_netlist,
    "tree": generate_tree_topology_netlist,
    "vertical_line": generate_vertical_line_topology_netlist,
    "horizontal_line": generate_horizontal_line_topology_netlist,
    "cnn": generate_cnn_topology_netlist,
}


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        prog="auto_gen_netlist",
        description="Script to auto-generate simple netlist topologies for AIE data flow",
    )
    argparser.add_argument(
        "--topology", 
        type=str, 
        dest="netlist_topologies",
        choices=list(TOPOLOGIES_CONVERSION.keys()),
        default="tree",
    )
    argparser.add_argument(
        "-nl",
        "--netlist", 
        type=str, 
        dest="output_netlist",
        default="AIE_data_flow_netlists/netlist.json",
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
