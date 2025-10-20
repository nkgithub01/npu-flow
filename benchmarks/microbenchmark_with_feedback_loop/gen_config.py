import yaml

data = {
    'name': "Microbenchmark with feedback loop - Verify placement effect on runtime",
    'clean': "make clean",
    'build': "make build",
    'compile': "make aiecc",
    'run': "make run",
    'tasks': []
}

tasklist = []
for enable_feedback in [0, 1]:
    for inout_size in [2**i for i in range(0, 21)]:
        taskname = f"placement_line_single_node{'_with_feedback' if enable_feedback else ''}_neighbour_inoutsize_{inout_size}_distance_1"
        data['tasks'].append({
            'name': taskname,
            'params': {
                'target_name':'line',
                'num_rows': 6,
                'num_cols': 8,
                'inout_size': inout_size,
                'expand_rate': 1000000,
                'placement': f'single_node_neighbour_distance_1',
                'random_seed': 0,
                'enable_feedback': enable_feedback
            },
            'output': "build/line.mlir"
        })
        tasklist.append(f"- microbenchmark_with_feedback_loop/{taskname}")

    for inout_size in [2**i for i in range(0, 12)]:
        for distance in range(1, 32):
            taskname = f"placement_line_single_node{'_with_feedback' if enable_feedback else ''}_inoutsize_{inout_size}_distance_{distance}"
            data['tasks'].append({
                'name': taskname,
                'params': {
                    'target_name':'line',
                    'num_rows': 6,
                    'num_cols': 8,
                    'inout_size': inout_size,
                    'expand_rate': 1000000,
                    'placement': f'single_node_distance_{distance}',
                    'random_seed': 0,
                    'enable_feedback': enable_feedback
                },
                'output': "build/line.mlir"
            })
            tasklist.append(f"- microbenchmark_with_feedback_loop/{taskname}")

    for length in range(4, 33):
        taskname = f"placement_line_regular_length_{length}{'_with_feedback' if enable_feedback else ''}"
        data['tasks'].append({
            'name': taskname,
            'params': {
                'target_name':'line',
                'num_rows': 6,
                'num_cols': 8,
                'inout_size': 256,
                'expand_rate': 100000,
                'placement': 'regular',
                'random_seed': 0,
                'length': length,
                'enable_feedback': enable_feedback
            },
            'output': "build/line.mlir"
        })
        tasklist.append(f"- microbenchmark_with_feedback_loop/{taskname}")
    
    for seed in range(10):
        for length in range(4,33):
            taskname = f"placement_line_random_length_{length}_seed_{seed}{'_with_feedback' if enable_feedback else ''}"
            data['tasks'].append({
                'name': taskname,
                'params': {
                    'target_name':'line',
                    'num_rows': 6,
                    'num_cols': 8,
                    'inout_size': 256,
                    'expand_rate': 100000,
                    'placement': 'random',
                    'random_seed': seed,
                    'length': length,
                    'enable_feedback': enable_feedback
                },
                'output': "build/line.mlir"
            })
            tasklist.append(f"- microbenchmark_with_feedback_loop/{taskname}")

for num_row in range(4, 7):
    for num_col in range(2, 9):
        taskname = f"placement_mesh_R{num_row}_C{num_col}"
        data['tasks'].append({
            'name': taskname,
            'params': {
                'target_name':'mesh',
                'num_rows': num_row,
                'num_cols': num_col,
                'inout_size': 10000,
                'random_seed': 0,
            },
            'output': "build/mesh.mlir"
        })
        tasklist.append(f"- microbenchmark_with_feedback_loop/{taskname}")

with open('config.yml', 'w') as file:
    yaml_content = yaml.dump(data, default_flow_style=False, sort_keys=False, indent=2)
    
    # Add extra indentation for task items to show they belong to tasks field
    import re
    lines = yaml_content.split('\n')
    output_lines = []
    
    for line in lines:
        if line.startswith('- name:'):
            # Add 2 spaces before the dash to indent task items
            output_lines.append('  ' + line)
        elif line.startswith('  ') and 'tasks:' not in line:
            # This handles all properties under each task item
            # Add 2 more spaces to align with indented task items
            output_lines.append('  ' + line)
        else:
            output_lines.append(line)
    
    # Add spaces between tasks
    final_lines = []
    first_task = True
    
    for i, line in enumerate(output_lines):
        # Add blank line before each task name (except the first one)
        if line.strip().startswith('- name:') and not first_task:
            final_lines.append('')
        
        final_lines.append(line)
        
        if line.strip().startswith('- name:'):
            first_task = False
    
    file.write('\n'.join(final_lines))

with open('tasklist.yml', 'w') as file:
    file.write('\n'.join(tasklist))