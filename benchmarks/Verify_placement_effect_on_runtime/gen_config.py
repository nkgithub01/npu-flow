import yaml

data = {
    'name': "verify Runtime with different placement",
    'clean': "make clean",
    'build': "make build",
    'compile': "make aiecc",
    'run': "make run",
    'tasks': []
}

tasklist = []
for with_feedback in [0, 1]:
    for inout_size in [2**i for i in range(0, 21)]:
        taskname = f"placement_single_node{'_with_feedback' if with_feedback else ''}_neighbour_inoutsize_{inout_size}_distance_1"
        data['tasks'].append({
            'name': taskname,
            'params': {
                'num_rows': 6,
                'num_cols': 8,
                'inout_size': inout_size,
                'expand_rate': 1000000,
                'placement': f'single_node_neighbour_distance_1',
                'placement_seed': 0,
                'enable_feedback': with_feedback
            },
            'output': "build/verify.mlir"
        })
        tasklist.append(f"- Verify_placement_effect_on_runtime/{taskname}")

    for inout_size in [2**i for i in range(0, 12)]:
        for distance in range(1, 32):
            taskname = f"placement_single_node{'_with_feedback' if with_feedback else ''}_inoutsize_{inout_size}_distance_{distance}"
            data['tasks'].append({
                'name': taskname,
                'params': {
                    'num_rows': 6,
                    'num_cols': 8,
                    'inout_size': inout_size,
                    'expand_rate': 1000000,
                    'placement': f'single_node_distance_{distance}',
                    'placement_seed': 0,
                    'enable_feedback': with_feedback
                },
                'output': "build/verify.mlir"
            })
            tasklist.append(f"- Verify_placement_effect_on_runtime/{taskname}")

    for placement in ["regular", "1-hop", "2-hop"]:
        taskname = f"placement_{placement}_with_feedback" if with_feedback else f"placement_{placement}"
        data['tasks'].append({
            'name': taskname,
            'params': {
                'num_rows': 6,
                'num_cols': 8,
                'inout_size': 256,
                'expand_rate': 1000000,
                'placement': placement,
                'placement_seed': 0,
                'enable_feedback': with_feedback
            },
            'output': "build/verify.mlir"
        })
        tasklist.append(f"- Verify_placement_effect_on_runtime/{taskname}")
    
    for seed in range(10):
        for length in range(4,33):
            taskname = f"length_{length}_random_seed_{seed}_with_feedback" if with_feedback else f"length_{length}_random_seed_{seed}"
            data['tasks'].append({
                'name': taskname,
                'params': {
                    'num_rows': 6,
                    'num_cols': 8,
                    'inout_size': 256,
                    'expand_rate': 100000,
                    'placement': 'random',
                    'placement_seed': seed,
                    'enable_feedback': with_feedback,
                    'length': length
                },
                'output': "build/verify.mlir"
            })
            tasklist.append(f"- Verify_placement_effect_on_runtime/{taskname}")

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