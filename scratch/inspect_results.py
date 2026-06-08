import pandas as pd
import numpy as np

csv_path = r"c:\Users\etxan\OneDrive\Documentos\TFM_final\output\results\results.csv"
df = pd.read_csv(csv_path)

print("Columns:")
print(list(df.columns))
print("\nUnique datasets:", df['dataset'].unique())
print("Unique solvers:", df['solver'].unique())
print("Unique N by dataset:")
for ds in df['dataset'].unique():
    print(f"  {ds}:", df[df['dataset'] == ds]['N'].unique())
print("Unique p by solver:")
for s in df['solver'].unique():
    print(f"  {s}:", df[df['solver'] == s]['p'].unique())
print("\nRow counts by dataset and solver:")
print(df.groupby(['dataset', 'solver']).size())
print("\nAny NaN values in key columns?")
for col in ['objective', 'gap', 'sharpe', 'feasible', 'runtime_seconds']:
    nan_count = df[col].isna().sum()
    print(f"  {col}: {nan_count} NaNs")
