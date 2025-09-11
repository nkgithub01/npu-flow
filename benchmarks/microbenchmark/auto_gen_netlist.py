import json
import argparse

def write_netlist_to_file(netlist, output_file):
    with open(output_file, 'w') as netlist_file:
        json.dump(netlist, netlist_file, indent=4)


def helper_create_node(netlist, col_x, row_y):
    if row_y == 0:
        node_type = "SHIM"
    elif row_y == 1:
        node_type = "MEM"
    else:
        node_type = "COMP"
    node_id = len(netlist["nodes"])
    netlist["nodes"].append({
        "id": node_id,
        "type": node_type,
        "col_x": col_x,
        "row_y": row_y,
    })
    return node_id


def helper_connect_nodes(args, netlist, src_node_id, dst_node_ids):
    net_id = len(netlist["nets"])
    netlist["nets"].append({
        "net_id": net_id,
        "src_id": src_node_id,
        "dst_id": dst_node_ids,
        "depths": [args.obj_fifo_depth],
        "byte_size_per_depth": args.obj_fifo_byte_size_per_depth,
    })
    return net_id


def helper_link_nets(netlist, link_src_net_ids, link_dst_net_ids):
    if not isinstance(link_src_net_ids, list):
        link_src_net_ids = [link_src_net_ids]
    if not isinstance(link_dst_net_ids, list):
        link_dst_net_ids = [link_dst_net_ids]

    netlist["links"].append({
        "src_net_ids": link_src_net_ids,
        "dst_net_ids": link_dst_net_ids
    })


def generate_single_multicast_netlist(args, netlist):
    # Define nodes
    nodes_loc2ID_lookup = dict()
    for x in range(1):
        for y in range(5):
            node_id = helper_create_node(netlist, x, y)
            nodes_loc2ID_lookup[(x, y)] = node_id

    # Define nets connections
    # The SHIM node is connected to the MEM node in the same column and linked to the net from MEM to the compute core 1
    # Create a net to the MEM node in the same column
    SHIM_id = nodes_loc2ID_lookup[(0, 0)]
    MEM_id = nodes_loc2ID_lookup[(0, 1)]
    COMP0_id = nodes_loc2ID_lookup[(0, 2)]
    COMP1_id = nodes_loc2ID_lookup[(0, 3)]
    COMP2_id = nodes_loc2ID_lookup[(0, 4)]
    link_src_net_id = helper_connect_nodes(args, netlist, SHIM_id, [MEM_id])
    # Link this net to the MEM node in the same column
    link_dst_net_id = helper_connect_nodes(args, netlist, MEM_id, [COMP1_id])
    helper_link_nets(netlist, [link_src_net_id], [link_dst_net_id])

    # The compute core 1 broadcasts to all other compute cores
    helper_connect_nodes(args, netlist, COMP1_id, [COMP0_id, COMP2_id])

    # The output of the compute core 0 and 2 are send back to the MEM then forward to SHIM
    link_src_net_id = helper_connect_nodes(args, netlist, COMP0_id, [MEM_id])
    # Link this net to the MEM node in the same column
    link_dst_net_id = helper_connect_nodes(args, netlist, MEM_id, [SHIM_id])
    helper_link_nets(netlist, [link_src_net_id], [link_dst_net_id])
    link_src_net_id = helper_connect_nodes(args, netlist, COMP2_id, [MEM_id])
    # Link this net to the MEM node in the same column
    link_dst_net_id = helper_connect_nodes(args, netlist, MEM_id, [SHIM_id])
    helper_link_nets(netlist, [link_src_net_id], [link_dst_net_id])

    return netlist


def generate_2d_mesh_topology_netlist(args, netlist):
    # Define nodes
    nodes_loc2ID_lookup = dict()
    for x in range(args.num_cols):
        for y in range(args.num_rows):
            node_id = helper_create_node(netlist, x, y)
            nodes_loc2ID_lookup[(x, y)] = node_id

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
                link_src_net_id = helper_connect_nodes(args, netlist, same_col_SHIM_id, [same_col_MEM_id])
                # Link this net to the MEM node in the same column
                link_dst_net_id = helper_connect_nodes(args, netlist, same_col_MEM_id, [same_col_COMP_id])
                helper_link_nets(netlist, [link_src_net_id], [link_dst_net_id])

            # The middle region compute core is connected to its right and top neighbors
            if x < args.num_cols - 1 and 1 < y < args.num_rows - 1:
                # Create a net to the right neighbor
                right_neighbor_id = nodes_loc2ID_lookup[(x + 1, y)]
                helper_connect_nodes(args, netlist, node_id, [right_neighbor_id])

                # Create a net to the top neighbor
                top_neighbor_id = nodes_loc2ID_lookup[(x, y + 1)]
                helper_connect_nodes(args, netlist, node_id, [top_neighbor_id])

            # The rightmost column nodes are connected to their top neighbor
            if x == args.num_cols - 1 and 1 < y < args.num_rows - 1:
                # Create a net to the top neighbor
                top_neighbor_id = nodes_loc2ID_lookup[(x, y + 1)]
                helper_connect_nodes(args, netlist, node_id, [top_neighbor_id])
            
            # The top row nodes are connected to their right neighbor
            if y == args.num_rows - 1 and x < args.num_cols - 1:
                # Create a net to the right neighbor
                right_neighbor_id = nodes_loc2ID_lookup[(x + 1, y)]
                helper_connect_nodes(args, netlist, node_id, [right_neighbor_id])

            # The top row nodes are also connected back to MEM node in the same column
            if y == args.num_rows - 1:
                # Create a net to the MEM node in the same column
                same_col_MEM_id = nodes_loc2ID_lookup[(x, 1)]
                link_src_net_id = helper_connect_nodes(args, netlist, node_id, [same_col_MEM_id])
                # Link this net to the SHIM node in the same column
                same_col_SHIM_id = nodes_loc2ID_lookup[(x, 0)]
                link_dst_net_id = helper_connect_nodes(args, netlist, same_col_MEM_id, [same_col_SHIM_id])
                helper_link_nets(netlist, [link_src_net_id], [link_dst_net_id])

    return netlist


def generate_3d_mesh_topology_netlist(args, netlist):
    # Define nodes
    # Create a 3D mesh with 3x3x3 compute nodes
    nodes_loc2ID_lookup = dict()
    for x in range(8):
        for y in range(6):
            # Skip unused nodes in 3D mesh shape
            if y == 5 and x <= 4:
                continue
            node_id = helper_create_node(netlist, x, y)
            nodes_loc2ID_lookup[(x, y)] = node_id

    # Define nets connections
    # In each level, each compute node is connected to its right, top, and next level neighbors
    # Level 1
    for x in range(0,3):
        for y in range(2,5):
            node_id = nodes_loc2ID_lookup[(x, y)]
            # Create a net to the right neighbor
            if x < 2:
                right_neighbor_id = nodes_loc2ID_lookup[(x + 1, y)]
                helper_connect_nodes(args, netlist, node_id, [right_neighbor_id])
            
            # Create a net to the top neighbor
            if y < 4:
                top_neighbor_id = nodes_loc2ID_lookup[(x, y + 1)]
                helper_connect_nodes(args, netlist, node_id, [top_neighbor_id])
            
            # Create a net to the next level neighbor
            next_level_neighbor_id = nodes_loc2ID_lookup[(x + 3, y)]
            helper_connect_nodes(args, netlist, node_id, [next_level_neighbor_id])
        
        # The input one is from SHIM node 0 to MEM node 0 to COMP nodes in level 1 
        if x == 0:
            SHIM_id = nodes_loc2ID_lookup[(x, 0)]
            MEM_id = nodes_loc2ID_lookup[(x, 1)]
            COMP_ids = [nodes_loc2ID_lookup[(x, y)] for y in range(2, 5)]
            link_src_net_id = helper_connect_nodes(args, netlist, SHIM_id, [MEM_id])
            link_dst_net_id = helper_connect_nodes(args, netlist, MEM_id, COMP_ids)
            helper_link_nets(netlist, [link_src_net_id], [link_dst_net_id])
        
        # The input two is from SHIM node in the same column to MEM node in the same column to COMP nodes in level 1
        SHIM_id = nodes_loc2ID_lookup[(x, 0)]
        MEM_id = nodes_loc2ID_lookup[(x, 1)]
        COMP_id = nodes_loc2ID_lookup[(x, 2)]
        link_src_net_id = helper_connect_nodes(args, netlist, SHIM_id, [MEM_id])
        link_dst_net_id = helper_connect_nodes(args, netlist, MEM_id, [COMP_id])
        helper_link_nets(netlist, [link_src_net_id], [link_dst_net_id])

    # Level 2
    for x in range(3,6):
        for y in range(2,5):
            node_id = nodes_loc2ID_lookup[(x, y)]
            # Create a net to the right neighbor
            if x < 5:
                right_neighbor_id = nodes_loc2ID_lookup[(x + 1, y)]
                helper_connect_nodes(args, netlist, node_id, [right_neighbor_id])
            
            # Create a net to the top neighbor
            if y < 4:
                top_neighbor_id = nodes_loc2ID_lookup[(x, y + 1)]
                helper_connect_nodes(args, netlist, node_id, [top_neighbor_id])
            
            # Create a net to the next level neighbor
            if x < 5:
                next_level_neighbor_id = nodes_loc2ID_lookup[(x + 3, y)]
            else:
                next_level_neighbor_id = nodes_loc2ID_lookup[(9-y, 5)]
            helper_connect_nodes(args, netlist, node_id, [next_level_neighbor_id])

        # The input one is from SHIM node 3 to MEM node 3 to COMP nodes in level 2
        if x == 3:
            SHIM_id = nodes_loc2ID_lookup[(x, 0)]
            MEM_id = nodes_loc2ID_lookup[(x, 1)]
            COMP_ids = [nodes_loc2ID_lookup[(x, y)] for y in range(2, 5)]
            link_src_net_id = helper_connect_nodes(args, netlist, SHIM_id, [MEM_id])
            link_dst_net_id = helper_connect_nodes(args, netlist, MEM_id, COMP_ids)
            helper_link_nets(netlist, [link_src_net_id], [link_dst_net_id])
    
    # Level 3
    for x in range(6,8):
        for y in range(2,5):
            node_id = nodes_loc2ID_lookup[(x, y)]
            # Create a net to the right neighbor
            if x < 7:
                right_neighbor_id = nodes_loc2ID_lookup[(x + 1, y)]
                helper_connect_nodes(args, netlist, node_id, [right_neighbor_id])
            elif x == 7:
                # The rightmost node in level 3 is arranged to the top of the array
                right_neighbor_id = nodes_loc2ID_lookup[(9-y, 5)]
                helper_connect_nodes(args, netlist, node_id, [right_neighbor_id])
            
            # Create a net to the top neighbor
            if y < 4:
                top_neighbor_id = nodes_loc2ID_lookup[(x, y + 1)]
                helper_connect_nodes(args, netlist, node_id, [top_neighbor_id])
            
        # The top nodes in the last level are connected back to MEM node in the same column
        node_id = nodes_loc2ID_lookup[(x, 4)]
        same_col_MEM_id = nodes_loc2ID_lookup[(x, 1)]
        link_src_net_id = helper_connect_nodes(args, netlist, node_id, [same_col_MEM_id])
        # Link this net to the SHIM node in the same column
        same_col_SHIM_id = nodes_loc2ID_lookup[(x, 0)]
        link_dst_net_id = helper_connect_nodes(args, netlist, same_col_MEM_id, [same_col_SHIM_id])
        helper_link_nets(netlist, [link_src_net_id], [link_dst_net_id])

        # The input one is from SHIM node 6 to MEM node 6 to COMP nodes in level 3
        if x == 6:
            SHIM_id = nodes_loc2ID_lookup[(x, 0)]
            MEM_id = nodes_loc2ID_lookup[(x, 1)]
            COMP_ids = [nodes_loc2ID_lookup[(x, y)] for y in range(2, 5)]
            link_src_net_id = helper_connect_nodes(args, netlist, SHIM_id, [MEM_id])
            link_dst_net_id = helper_connect_nodes(args, netlist, MEM_id, COMP_ids)
            helper_link_nets(netlist, [link_src_net_id], [link_dst_net_id])
    
    for x in range(5,8):
        y = 5
        node_id = nodes_loc2ID_lookup[(x, y)]

        # The top nodes in the last level are connected back to MEM node in the same column
        if x == 5:
            same_col_MEM_id = nodes_loc2ID_lookup[(x, 1)]
            link_src_net_id = helper_connect_nodes(args, netlist, node_id, [same_col_MEM_id])
            # Link this net to the SHIM node in the same column
            same_col_SHIM_id = nodes_loc2ID_lookup[(x, 0)]
            link_dst_net_id = helper_connect_nodes(args, netlist, same_col_MEM_id, [same_col_SHIM_id])
            helper_link_nets(netlist, [link_src_net_id], [link_dst_net_id])
        else:
            # Create a net to the top neighbor
            top_neighbor_id = nodes_loc2ID_lookup[(x-1, y)]
            helper_connect_nodes(args, netlist, node_id, [top_neighbor_id])

    return netlist


def generate_line_topology_netlist(args, netlist):
    # Define nodes
    nodes_loc2ID_lookup = dict()
    for x in range(args.num_cols):
        for y in range(args.num_rows):
            node_id = helper_create_node(netlist, x, y)
            nodes_loc2ID_lookup[(x, y)] = node_id

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
                link_src_net_id = helper_connect_nodes(args, netlist, same_col_SHIM_id, [same_col_MEM_id])
                # Link this net to the MEM node in the same column
                link_dst_net_id = helper_connect_nodes(args, netlist, same_col_MEM_id, [same_col_COMP_id])
                helper_link_nets(netlist, [link_src_net_id], [link_dst_net_id])

            # The middle region compute core is connected to its top neighbors
            elif 1 < y < args.num_rows - 1:
                # Create a net to the top neighbor
                top_neighbor_id = nodes_loc2ID_lookup[(x, y + 1)]
                helper_connect_nodes(args, netlist, node_id, [top_neighbor_id])
            
            # The top row nodes are also connected back to MEM node in the same column
            elif y == args.num_rows - 1:
                # Create a net to the MEM node in the same column
                same_col_MEM_id = nodes_loc2ID_lookup[(x, 1)]
                link_src_net_id = helper_connect_nodes(args, netlist, node_id, [same_col_MEM_id])
                # Link this net to the SHIM node in the same column
                same_col_SHIM_id = nodes_loc2ID_lookup[(x, 0)]
                link_dst_net_id = helper_connect_nodes(args, netlist, same_col_MEM_id, [same_col_SHIM_id])
                helper_link_nets(netlist, [link_src_net_id], [link_dst_net_id])

    return netlist

def generate_tree_topology_netlist(args, netlist):
    # Define nodes
    nodes_loc2ID_lookup = dict()
    for x in range(args.num_cols):
        for y in range(args.num_rows):
            node_id = helper_create_node(netlist, x, y)
            nodes_loc2ID_lookup[(x, y)] = node_id

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
            link_src_net_id = helper_connect_nodes(args, netlist, same_col_SHIM_id, [same_col_MEM_id])
            # Link this net to the SHIM node in the same column
            link_dst_net_id = helper_connect_nodes(args, netlist, same_col_MEM_id, [node_id])
            helper_link_nets(netlist, [link_src_net_id], [link_dst_net_id])

        has_no_children = True
        # Connect to left child
        left_child_idx = idx * 2 + 1
        left_child_x = left_child_idx % args.num_cols
        left_child_y = 2 + (left_child_idx // args.num_cols)
        if left_child_x < args.num_cols and left_child_y < args.num_rows:
            left_child_id = nodes_loc2ID_lookup[(left_child_x, left_child_y)]
            helper_connect_nodes(args, netlist, node_id, [left_child_id])
            has_no_children = False
        
        # Connect to right child
        right_child_idx = idx * 2 + 2
        right_child_x = right_child_idx % args.num_cols
        right_child_y = 2 + (right_child_idx // args.num_cols)
        if right_child_x < args.num_cols and right_child_y < args.num_rows:
            right_child_id = nodes_loc2ID_lookup[(right_child_x, right_child_y)]
            helper_connect_nodes(args, netlist, node_id, [right_child_id])
            has_no_children = False
        
        if has_no_children:
            # If the node has no children, connect it to the MEM node in the same column
            same_col_MEM_id = nodes_loc2ID_lookup[(x, 1)]
            link_src_net_id = helper_connect_nodes(args, netlist, node_id, [same_col_MEM_id])
            # Link this net to the SHIM node in the same column
            same_col_SHIM_id = nodes_loc2ID_lookup[(x, 0)]
            link_dst_net_id = helper_connect_nodes(args, netlist, same_col_MEM_id, [same_col_SHIM_id])
            helper_link_nets(netlist, [link_src_net_id], [link_dst_net_id])

    return netlist


def generate_cnn_topology_netlist(args, netlist):
    # Define nodes
    nodes_loc2ID_lookup = dict()
    for x in range(args.num_cols):
        for y in range(args.num_rows):
            node_id = helper_create_node(netlist, x, y)
            nodes_loc2ID_lookup[(x, y)] = node_id

    # Define nets connections
    # Input X is connected from SHIM node 0, 1, 2, 3 to MEM node 0, 1, 2, 3 to COMP nodes in the first column
    for idx in range(4):
        # Connect SHIM node to MEM node in the same column
        x = idx
        SHIM_id = nodes_loc2ID_lookup[(x, 0)]
        MEM_id = nodes_loc2ID_lookup[(x, 1)]
        link_src_net_id = helper_connect_nodes(args, netlist, SHIM_id, [MEM_id])
        
        # Link this net to the Comp node in the first column
        y = 5 - idx
        COMP_id = nodes_loc2ID_lookup[(0, y)]
        link_dst_net_id = helper_connect_nodes(args, netlist, MEM_id, [COMP_id])
        helper_link_nets(netlist, [link_src_net_id], [link_dst_net_id])

    # Input W is connected from SHIM node 0 to MEM node 0 to COMP nodes in the first row
    for idx in range(1):
        # Connect SHIM node to MEM node in the same column
        x = idx
        SHIM_id = nodes_loc2ID_lookup[(x, 0)]
        MEM_id = nodes_loc2ID_lookup[(x, 1)]
        link_src_net_id = helper_connect_nodes(args, netlist, SHIM_id, [MEM_id])

        # Link this net to the Comp node in the first column
        COMP_ids = [nodes_loc2ID_lookup[(0, y)] for y in range(2, 6)]
        link_dst_net_id = helper_connect_nodes(args, netlist, MEM_id, COMP_ids)
        helper_link_nets(netlist, [link_src_net_id], [link_dst_net_id])

    # The first column compute nodes are connected to the second column compute nodes
    for y in range(2, 6):
        # Get the node ID for the compute nodes
        first_col_COMP_id = nodes_loc2ID_lookup[(0, y)]
        second_col_COMP_id = nodes_loc2ID_lookup[(1, y)]
        
        # Create a net to connect the two compute nodes
        helper_connect_nodes(args, netlist, first_col_COMP_id, [second_col_COMP_id])

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
        helper_connect_nodes(args, netlist, second_col_COMP_id, [third_col_COMP_id])

    # The third column compute nodes are connected in a way that row_5 to row_4, row_3 to row_2. row_4 and row_2 are connected to the MEM node in the same column
    # The connection to the MEM is forwarded to the COMP node in column 3, 4, 5, 6
    linked_src_net_ids = []
    for y in range(2, 6):
        # Get the node ID for the compute nodes in the third column
        third_col_COMP_id = nodes_loc2ID_lookup[(2, y)]
        
        if y == 5 or y == 3:
            # Connect to the Comp node below in the same column
            below_COMP_id = nodes_loc2ID_lookup[(2, y-1)]
            helper_connect_nodes(args, netlist, third_col_COMP_id, [below_COMP_id])

        elif y == 4 or y == 2:
            # Connect to the MEM node in the same column
            same_col_MEM_id = nodes_loc2ID_lookup[(2, 1)]
            net_id = helper_connect_nodes(args, netlist, third_col_COMP_id, [same_col_MEM_id])
            linked_src_net_ids.append(net_id)
    # Link this net to the net to the COMP nodes in columns 3, 4, 5, 6
    for idx in range(1):
        # Get the node ID for the compute nodes in columns 3, 4, 5, 6
        COMP_ids = [nodes_loc2ID_lookup[(x, y)] for x in range(3, 7) for y in range(2, 6)]
        # Connect the net from MEM nodes in column 1 to the COMP nodes in columns 3, 4, 5, 6
        MEM_id = nodes_loc2ID_lookup[(2, 1)]
        link_dst_net_id = helper_connect_nodes(args, netlist, MEM_id, COMP_ids)
        helper_link_nets(netlist, linked_src_net_ids, [link_dst_net_id])

    # The weight for Fully connected layers is sent in from SHIM nodes 2, 3, 4, 5 to MEM nodes 2, 3, 4, 5 to COMP nodes in columns 3, 4, 5, 6
    for x in range(2, 6):
        # Connect SHIM node to MEM node in the same column
        SHIM_id = nodes_loc2ID_lookup[(x, 0)]
        MEM_id = nodes_loc2ID_lookup[(x, 1)]
        link_src_net_id = helper_connect_nodes(args, netlist, SHIM_id, [MEM_id])

        # Link this net to the Comp nodes in columns 3, 4, 5, 6
        COMP_ids = [nodes_loc2ID_lookup[(x+1, y)] for y in range(2, 6)]
        link_dst_net_id = helper_connect_nodes(args, netlist, MEM_id, COMP_ids)
        helper_link_nets(netlist, [link_src_net_id], [link_dst_net_id])

    # The output from the compute nodes in columns 3, 4, 5, 6 are chained sequentially to the MEM node in column 7
    linked_src_net_ids = []
    for x in range(3, 7):
        for y in range(2, 6):
            # Get the node ID for the compute nodes in columns 3, 4, 5, 6
            COMP_id = nodes_loc2ID_lookup[(x, y)]

            if y == 2 and x == 6:
                # Connect to the MEM node
                MEM_id = nodes_loc2ID_lookup[(x, 1)]
                net_id = helper_connect_nodes(args, netlist, COMP_id, [MEM_id])
                linked_src_net_ids.append(net_id)
            elif y == 2:
                # Connect to the COMP node to the right in the same row
                right_COMP_id = nodes_loc2ID_lookup[(x+1, y)]
                helper_connect_nodes(args, netlist, COMP_id, [right_COMP_id])
            else:
                # Connect to the COMP node below in the same column
                below_COMP_id = nodes_loc2ID_lookup[(x, y-1)]
                helper_connect_nodes(args, netlist, COMP_id, [below_COMP_id])

    # Link this net to the net that sends data from the MEM node in column 6 to the COMP node in column 7
    for idx in range(1):
        # Get the node ID for the compute nodes in column 7
        COMP_ids = [nodes_loc2ID_lookup[(7, y)] for y in range(2, 6)]
        # Connect the net from MEM node in column 6 to the COMP node in column 7
        MEM_id = nodes_loc2ID_lookup[(6, 1)]
        link_dst_net_id = helper_connect_nodes(args, netlist, MEM_id, COMP_ids)
        helper_link_nets(netlist, linked_src_net_ids, [link_dst_net_id])

    # The weight for the final Fully connected layer is sent in from SHIM node 6 to MEM node 6 to COMP nodes in column 7
    for idx in range(1):
        # Connect SHIM node to MEM node in the same column
        SHIM_id = nodes_loc2ID_lookup[(6, 0)]
        MEM_id = nodes_loc2ID_lookup[(6, 1)]
        link_src_net_id = helper_connect_nodes(args, netlist, SHIM_id, [MEM_id])

        # Link this net to the Comp nodes in column 7
        COMP_ids = [nodes_loc2ID_lookup[(7, y)] for y in range(2, 6)]
        link_dst_net_id = helper_connect_nodes(args, netlist, MEM_id, COMP_ids)
        helper_link_nets(netlist, [link_src_net_id], [link_dst_net_id])

    # The output of the COMP nodes in column 7 are sent to the COMP node below it. The output of COMP node at row 2 is sent to the MEM node then SHIM node in column 7
    for y in range(2, 5):
        # the output of the top 3 COMP nodes in column 7 are sent to the COMP node below it
        if y > 2:
            # Get the node ID for the compute nodes in column 7
            COMP_id1 = nodes_loc2ID_lookup[(7, y)]
            # Connect to the COMP node at row 2 in the same column
            COMP_id2 = nodes_loc2ID_lookup[(7, y-1)]
            helper_connect_nodes(args, netlist, COMP_id1, [COMP_id2])
        
        # the output of COMP node at row 2 is sent to the MEM node then SHIM node in column 7
        elif y == 2:
            # Get the node ID for the compute nodes in column 7
            COMP_id = nodes_loc2ID_lookup[(7, y)]
            # Connect to the MEM node in the same column
            MEM_id = nodes_loc2ID_lookup[(7, 1)]
            link_src_net_id = helper_connect_nodes(args, netlist, COMP_id, [MEM_id])
            # Link this net to the SHIM node in the same column
            SHIM_id = nodes_loc2ID_lookup[(7, 0)]
            link_dst_net_id = helper_connect_nodes(args, netlist, MEM_id, [SHIM_id])
            helper_link_nets(netlist, [link_src_net_id], [link_dst_net_id])

    return netlist


def generate_netlist(args):
    netlist = dict(
        nodes = [],
        nets = [],
        links = []
    )

    if args.netlist_topologies not in TOPOLOGIES_CONVERSION:
        raise ValueError(f"Unsupported netlist topology: {args.netlist_topologies}. Supported topologies: {list(TOPOLOGIES_CONVERSION.keys())}")
    generate_func = TOPOLOGIES_CONVERSION[args.netlist_topologies]
    netlist = generate_func(args, netlist)

    return netlist


def main(args):

    # Generate netlist for AIE data flow
    netlist = generate_netlist(args)

    # Write the output netlist
    write_netlist_to_file(netlist, args.output_netlist)


TOPOLOGIES_CONVERSION = {
    "single_multicast": generate_single_multicast_netlist,
    "2d_mesh": generate_2d_mesh_topology_netlist,
    "3d_mesh": generate_3d_mesh_topology_netlist,
    "line": generate_line_topology_netlist,
    "tree": generate_tree_topology_netlist,
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
        choices=TOPOLOGIES_CONVERSION.keys(),
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
