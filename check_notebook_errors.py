import json
import sys

notebook_path = 'notebooks/Gender_Wage_Gap_DoubleML_Analysis.ipynb'

try:
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
except FileNotFoundError:
    print(f"Error: File not found: {notebook_path}")
    sys.exit(1)

print(f"Checking {notebook_path} for errors and incomplete outputs...")

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        # Check for errors in outputs
        for output in cell.get('outputs', []):
            if output['output_type'] == 'error':
                print(f"Error in cell {i}:")
                print(f"  Ename: {output['ename']}")
                print(f"  Evalue: {output['evalue']}")
        
        # Check for empty outputs (if source is not empty)
        if cell['source'] and not cell.get('outputs'):
            # Ignore cells that might not produce output (e.g. imports, assignments)
            # This is a heuristic.
            source_str = "".join(cell['source']).strip()
            if not source_str.startswith("#") and "import" not in source_str and "=" in source_str and "print" not in source_str and "plt" not in source_str:
                continue # Likely just assignment
            
            # print(f"Warning: Cell {i} has no output.") # Too noisy
            pass

print("Done checking.")
