import pandas as pd
from sklearn.preprocessing import StandardScaler

print("Successful!")

# Porosity Interpolation

df = pd.read_excel("porosity_ANN_dataset.xlsx")

print("Missing values before interpolation:")
print(df.isna().sum())

df_interpolated = df.interpolate(method='linear')

print("\nMissing values after interpolation:")
print(df_interpolated.isna().sum())

print("\nSample rows after interpolation:")
print(df_interpolated.head(10))

df_interpolated.to_excel("Porosity_interpolation_result.xlsx", index=False)

# Permeability interpolation

df = pd.read_excel("permeability_ANN_dataset.xlsx")

print("Missing values before interpolation:")
print(df.isna().sum())

df_interpolated = df.interpolate(method='linear')

print("\nMissing values after interpolation:")
print(df_interpolated.isna().sum())

print("\nSample rows after interpolation:")
print(df_interpolated.head(10))

df_interpolated.to_excel("Permeability_interpolation_result.xlsx", index=False)

# For Porosity Standardization

df = pd.read_excel('porosity_ANN_dataset.xlsx')
df = df.interpolate(method='linear')

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
if 'DEPTH' in numeric_cols:
    numeric_cols = numeric_cols.drop('DEPTH')

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

df.to_excel('Porosity_Preprocessed_Data.xlsx', index=False)
print("Successful!")

# For Permeability Standardization

df = pd.read_excel('permeability_ANN_dataset.xlsx')
df = df.interpolate(method='linear')

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
if 'DEPTH' in numeric_cols:
    numeric_cols = numeric_cols.drop('DEPTH')

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

df.to_excel('Permeability_Preprocessed_Data.xlsx', index=False)

# Porosity Z-Score Normalisation Descriptive Statistics

features = ['CALI', 'DRHO', 'DT', 'GR', 'NPHI', 'PEF', 'RHOB']
X = df_interpolated[features]

print("Before Scaling:")
print(X.describe().T[['min', 'max', 'mean', 'std']])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=features)

print("\nAfter Scaling:")
print(X_scaled_df.describe().T[['min', 'max', 'mean', 'std']])

# Permeability Z-Score Normalisation Descriptive Statistics

features = ['CALI', 'DRHO', 'DT', 'GR', 'NPHI', 'PEF', 'RHOB', 'RM', 'RT', 'RD']
X = df_interpolated[features]

print("Before Scaling:")
print(X.describe().T[['min', 'max', 'mean', 'std']])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=features)

print("\nAfter Scaling:")
print(X_scaled_df.describe().T[['min', 'max', 'mean', 'std']])

