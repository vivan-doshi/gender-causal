import json
import sys

notebook_path = 'notebooks/Gender_Wage_Gap_DoubleML_Analysis.ipynb'

try:
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
except FileNotFoundError:
    print(f"Error: File not found: {notebook_path}")
    sys.exit(1)

print(f"Last 3 cells of {notebook_path}:")

cells = nb['cells']
for i, cell in enumerate(cells[-3:]):
    print(f"--- Cell {len(cells)-3+i} ---")
    print(f"Type: {cell['cell_type']}")
    print(f"Source: {cell['source']}")
    if cell['cell_type'] == 'code':
        print(f"Outputs: {cell.get('outputs', [])}")
