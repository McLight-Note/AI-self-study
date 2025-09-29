import pandas as pd

# Data Cleaning

df = pd.read_csv('pokemon.csv')

df = df.drop(columns=['Sp. Atk'])

# Handle missing data
df = df.dropna(subset=['Type 2'])
df = df.fillna({'Type 2': 'None'})

# Fix inconsistent values
df['Type 1'] = df['Type 1'].replace({"Grass": "GRASS",
                                     "Fire": "FIRE",
                                     'Water': "WATER"})

# Standardize text
df['Name'] = df['Name'].str.lower()

# Fix data types
df["Legendary"] = df['Legendary'].astype(bool)

# Remove the duplicates
df = df.drop_duplicates()

print(df.to_string())