import pandas as pd

# For Porosity input Data

filtered_data_1="input_data.xlsx"
sheet_name="Sheet1"
columns=["RHOB", "NPHI", "DT", "DTS", "GR", "NBGRCFM", "PEF", "CALI", "DRHO", "DEPTH"]

df=pd.read_excel('input_data.xlsx', 'Sheet1', usecols=columns, )


porosity_data = df[(df['DEPTH'] >= 3226.9) & (df['DEPTH'] <= 3441.9)]


porosity_data.to_excel('porosity_input_data.xlsx', index=False)
print("successful")

# For Permeability input Data

filtered_data_2="input_data.xlsx"
sheet_name="Sheet1"
columns=["RHOB", "NPHI", "DT", "DTS", "GR", "NBGRCFM", "PEF", "CALI", "DRHO", "RD", "RM", "RT", "DEPTH"]

df=pd.read_excel('input_data.xlsx', 'Sheet1', usecols=columns, )


permeability_data = df[(df['DEPTH'] >= 3226.9) & (df['DEPTH'] <= 3441.9)]


permeability_data.to_excel('permeability_input_data.xlsx', index=False)
print("successful")

# For Porosity Output Data

filtered_data_3="output_data.xlsx"
sheet_name="Sheet1"
column=["DEPTH", "PHIF"]

df=pd.read_excel('output_data.xlsx', 'Sheet1', usecols=column, )


porosity_output_data = df[(df['DEPTH'] >= 3226.9) & (df['DEPTH'] <= 3441.9)]


porosity_output_data.to_excel('porosity_output_data.xlsx', index=False)
print("successful")

# For Permeability Output Data

filtered_data_4="output_data.xlsx"
sheet_name="Sheet1"
column=["DEPTH", "KLOGH"]

df=pd.read_excel('output_data.xlsx', 'Sheet1', usecols=column, )


permeability_output_data = df[(df['DEPTH'] >= 3226.9) & (df['DEPTH'] <= 3441.9)]


permeability_output_data.to_excel('permeability_output_data.xlsx', index=False)
print("successful")

# Merging Porosity Data - Input Features and Output Target 

input_df = pd.read_excel('porosity_input_data.xlsx', sheet_name='Sheet1')
output_df = pd.read_excel('porosity_output_data.xlsx', sheet_name='Sheet1')


merged_df = pd.concat([input_df, output_df], axis=1)

merged_df.to_excel('porosity_ANN_dataset.xlsx', index=False)

print("Merging complete!")

# Merging Permeability Data - Input Features and Output Target

input_df = pd.read_excel('permeability_input_data.xlsx', sheet_name='Sheet1')
output_df = pd.read_excel('permeability_output_data.xlsx', sheet_name='Sheet1')


merged_df = pd.concat([input_df, output_df], axis=1)

merged_df.to_excel('permeability_ANN_dataset.xlsx', index=False)

