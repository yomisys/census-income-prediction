import pandas as pd

df = pd.read_csv("data/census.csv")

# Remove spaces from column names
df.columns = df.columns.str.strip()

# Remove spaces from string values
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].str.strip()

df.to_csv("data/census_clean.csv", index=False)
