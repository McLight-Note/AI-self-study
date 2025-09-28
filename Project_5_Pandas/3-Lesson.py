import pandas as pd

# Read CSV
df = pd.read_csv("pokemon.csv")
df2 = pd.read_json("pokemon.json")

print(df, df2)
print(df.to_string(), df2.to_string())

# Convert to JSON
'''
df.to_json("pokemon.json", orient="records", lines=False, indent=4)
'''