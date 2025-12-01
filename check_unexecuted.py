import json
import sys

notebook_path = 'notebooks/Gender_Wage_Gap_DoubleML_Analysis.ipynb'

try:
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
except FileNotFoundError:
    print(f"Error: File not found: {notebook_path}")
    sys.exit(1)

print(f"Checking {notebook_path} for unexecuted cells...")

unexecuted_count = 0
total_code_cells = 0

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        total_code_cells += 1
        if cell['execution_count'] is None and cell['source']:
            print(f"Cell {i} is unexecuted (execution_count is null). Source snippet: {cell['source'][:2]}")
            unexecuted_count += 1

print(f"Total code cells: {total_code_cells}")
print(f"Unexecuted code cells: {unexecuted_count}")
