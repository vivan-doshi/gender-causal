import json
import sys
import re

def update_notebook(notebook_path, script_path, output_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    with open(script_path, 'r', encoding='utf-8') as f:
        script_content = f.read()
    
    # Split script by "# %%"
    # The first chunk might be empty or imports if no cell marker at start
    # The script from extract_notebook_code.py adds "# %% \n" at start of each cell
    
    # We need to be careful. The extract script put "# %% \n" before each cell.
    # So we can split by "# %% \n" or just "# %%"
    
    cells_code = re.split(r'# %%\s*\n', script_content)
    
    # Remove the first empty chunk if it exists (before first marker)
    if cells_code and not cells_code[0].strip():
        cells_code.pop(0)
        
    # Now we iterate through notebook cells and replace code cells
    code_cell_idx = 0
    
    new_cells = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            if code_cell_idx < len(cells_code):
                # Update source
                # Remove the extra newlines we might have added or not
                source = cells_code[code_cell_idx]
                # If the source ends with multiple newlines, trim them but keep one?
                # Jupyter usually expects list of strings
                source_lines = source.splitlines(keepends=True)
                # Remove the first line if it's empty (artifact of split)
                if source_lines and source_lines[0].strip() == "":
                    source_lines.pop(0)
                
                cell['source'] = source_lines
                code_cell_idx += 1
        new_cells.append(cell)
    
    nb['cells'] = new_cells
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python update_notebook_code.py <original_notebook> <script_path> <output_notebook>")
        sys.exit(1)
    
    update_notebook(sys.argv[1], sys.argv[2], sys.argv[3])
