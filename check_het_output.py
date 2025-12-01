import json
import sys

notebook_path = 'notebooks/Gender_Wage_Gap_DoubleML_Analysis.ipynb'

try:
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
except FileNotFoundError:
    print(f"Error: File not found: {notebook_path}")
    sys.exit(1)

print(f"Checking outputs for Heterogeneous Effects in {notebook_path}...")

found_section = False
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "estimate_gap_subgroup" in source and "education_results" in source:
            found_section = True
            print(f"--- Cell {i} (Education Analysis) ---")
            for output in cell.get('outputs', []):
                if output['output_type'] == 'stream':
                    print(output['text'])
        
        if "age_results" in source:
            print(f"--- Cell {i} (Age Analysis) ---")
            for output in cell.get('outputs', []):
                if output['output_type'] == 'stream':
                    print(output['text'])

if not found_section:
    print("Could not find the Heterogeneous Effects code section.")
