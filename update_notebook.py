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
            # Data loading paths
            if "pd.read_excel('../data/raw/sample of sample ACS.xlsx')" in line or "pd.read_excel('/mnt/user-data/uploads/sample_of_sample_ACS.xlsx')" in line:
                new_source.append("acs_df = pd.read_excel('../data/raw/sample of sample ACS.xlsx')\n")
                modified = True
            elif "pd.read_csv('../data/raw/OCCSOC_Codes.csv', encoding='latin1')" in line or "pd.read_excel('/mnt/user-data/uploads/OCCSOC_Codes_cleaned.xlsx')" in line:
                new_source.append("occ_codes = pd.read_csv('../data/raw/OCCSOC_Codes.csv', encoding='latin1')\n")
                modified = True
            elif "pd.read_csv('../data/raw/IND_Codes.csv', encoding='latin1')" in line or "pd.read_excel('/mnt/user-data/uploads/IND_Codes.xlsx', header=None)" in line:
                new_source.append("ind_codes_raw = pd.read_csv('../data/raw/IND_Codes.csv', encoding='latin1')\n")
                modified = True
            # Fix hardcoded output paths
            elif "/home/claude/" in line:
                new_line = line.replace("/home/claude/", "./")
                new_source.append(new_line)
                modified = True
            # Fix object dtype error (exclude category columns)
            elif "region_cols = [col for col in df_model.columns if col.startswith('REGION_')]" in line:
                new_source.append("region_cols = [col for col in df_model.columns if col.startswith('REGION_') and col != 'REGION']\n")
                modified = True
            elif "occ_cols = [col for col in df_model.columns if col.startswith('OCC_')]" in line:
                new_source.append("occ_cols = [col for col in df_model.columns if col.startswith('OCC_') and col != 'OCC_CATEGORY']\n")
                modified = True
            elif "ind_cols = [col for col in df_model.columns if col.startswith('IND_')]" in line:
                new_source.append("ind_cols = [col for col in df_model.columns if col.startswith('IND_') and col != 'IND_CATEGORY']\n")
                modified = True
            # Inject robust filtering
            elif "control_features = continuous_features + binary_features + region_cols + occ_cols + ind_cols" in line:
                new_source.append(line)
                new_source.append("control_features = [col for col in control_features if pd.api.types.is_numeric_dtype(df_model[col])]\n")
                new_source.append("print(f'Filtered control features to {len(control_features)} numeric columns')\n")
                modified = True
            # Inject numeric conversion
            elif "df = acs_df.copy()" in line:
                new_source.append(line)
                new_source.append("df['INCWAGE'] = pd.to_numeric(df['INCWAGE'], errors='coerce')\n")
                new_source.append("df['PERWT'] = pd.to_numeric(df['PERWT'], errors='coerce')\n")
                new_source.append("df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce')\n")
                new_source.append("df['WKSWORK1'] = pd.to_numeric(df['WKSWORK1'], errors='coerce')\n")
                new_source.append("df['UHRSWORK'] = pd.to_numeric(df['UHRSWORK'], errors='coerce')\n")
                modified = True
            # Inject nuclear option before Model 4
            elif "# Model 4: Full model with occupation and industry controls" in line:
                new_source.append(line)
                new_source.append("print('Forcing numeric conversion on all model features...')\n")
                new_source.append("for col in ['FEMALE'] + control_features:\n")
                new_source.append("    df_final[col] = pd.to_numeric(df_final[col], errors='coerce')\n")
                new_source.append("df_final = df_final.dropna(subset=['FEMALE'] + control_features)\n")
                new_source.append("print('Done forcing numeric conversion.')\n")
                modified = True
            # Inject astype(float)
            elif "X4 = sm.add_constant(df_final[['FEMALE'] + control_features])" in line:
                new_source.append(line)
                new_source.append("X4 = X4.astype(float)\n")
                new_source.append("y = y.astype(float)\n")
                new_source.append("weights = weights.astype(float)\n")
                modified = True
            # Remove dml_procedure argument
            elif "dml_procedure='dml2'" in line:
                # Skip this line
                modified = True
                continue
            else:
                new_source.append(line)
        
        if modified:
            cell['source'] = new_source
            print("Updated cell.")

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated.")
