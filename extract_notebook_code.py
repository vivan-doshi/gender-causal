import json
import sys

def extract_code(notebook_path, output_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                f.write(f"# %% \n{source}\n\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_notebook_code.py <notebook_path> <output_path>")
        sys.exit(1)
    
    extract_code(sys.argv[1], sys.argv[2])
