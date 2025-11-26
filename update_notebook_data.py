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
            # Replace data loading
            if "acs_df = pd.read_excel('../data/raw/sample of sample ACS.xlsx')" in line:
                new_source.append("# Load full dataset\n")
                new_source.append("acs_df = pd.read_csv('../data/raw/Sample ACS 2021 for LMU.csv', encoding='latin1', dtype={'OCCSOC': str, 'IND': str})\n")
                modified = True
            elif "sample of sample ACS.xlsx" in line:
                 # Catch any other references (e.g. comments)
                 new_source.append(line.replace("sample of sample ACS.xlsx", "Sample ACS 2021 for LMU.csv"))
                 modified = True
            else:
                new_source.append(line)
        
        if modified:
            cell['source'] = new_source
            print("Updated data loading in cell.")

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated.")
