import json

notebook_path = 'notebooks/Gender_Wage_Gap_DoubleML_Analysis.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Iterate through cells to find the data loading cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        modified = False
        for line in source:
            # Replace image save paths
            if "plt.savefig('./" in line or "plt.savefig('" in line:
                # We want to replace './filename.png' with '../reports/figures/filename.png'
                # But we need to be careful not to break other things.
                # The grep showed they are mostly "./filename.png"
                if "./" in line and ".png" in line:
                    new_line = line.replace("./", "../reports/figures/")
                    new_source.append(new_line)
                    modified = True
                elif ".png" in line and "plt.savefig" in line and "/" not in line.split("'")[1]: # Case: plt.savefig('filename.png')
                     # This is a bit risky if logic is complex, but let's try simple replacement
                     # Actually, the grep showed they all use ./
                     new_source.append(line) 
                else:
                     new_source.append(line)
            # Replace csv save paths
            elif ".to_csv('./" in line:
                new_line = line.replace("./", "../data/processed/")
                new_source.append(new_line)
                modified = True
            else:
                new_source.append(line)
        
        if modified:
            cell['source'] = new_source
            print("Updated output paths in cell.")

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated.")
