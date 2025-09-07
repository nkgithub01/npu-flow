import yaml

data = {
    'name': "Microbenchmark Suite",
    'clean': "make clean",
    'build': "make",
    'run': "make run",
    'tasks': []
}

test_names = [
    '2d_mesh',
    'tree',
    'line'
]

tasklist = []

for test_name in test_names:
    for row in range(3,7):
        for col in range(1,9):
            data['tasks'].append({
                'name': f"{test_name}_R{row}_C{col}",
                'params': {
                    'netlist_topology': test_name,
                    'num_rows': row,
                    'num_cols': col
                },
                'output': "build/microbenchmark.mlir"
            })
            tasklist.append(f"- microbenchmark/{test_name}_R{row}_C{col}")

# Custom CNN only has 1 configuration
data['tasks'].append({
    'name': "Custom_CNN",
    'params': {
        'netlist_topology': "cnn",
        'num_rows': 6,
        'num_cols': 8
    },
    'output': "build/microbenchmark.mlir"
})
tasklist.append(f"- microbenchmark/Custom_CNN")

# toy example (single_multicast) only has 1 configuration
data['tasks'].append({
    'name': "single_multicast",
    'params': {
        'netlist_topology': "single_multicast",
        'num_rows': 6,
        'num_cols': 8
    },
    'output': "build/microbenchmark.mlir"
})
tasklist.append(f"- microbenchmark/single_multicast")

# 3d mesh only has 1 configuration
data['tasks'].append({
    'name': "3d_mesh",
    'params': {
        'netlist_topology': "3d_mesh",
        'num_rows': 6,
        'num_cols': 8
    },
    'output': "build/microbenchmark.mlir"
})
tasklist.append(f"- microbenchmark/3d_mesh")

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