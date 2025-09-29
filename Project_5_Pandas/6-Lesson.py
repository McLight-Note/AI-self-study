import pandas as pd

# aggregate functions

df = pd.read_csv('pokemon.csv')

# Whole DataFrame
print(df.mean(numeric_only=1))
print(df.sum(numeric_only=1))
print(df.min(numeric_only=1))
print(df.max(numeric_only=1))
print(df.count(numeric_only=1))

# Single column
print(df['HP'].mean())
print(df['HP'].sum())
print(df['HP'].min())
print(df['HP'].max())
print(df['HP'].count())

group = df.groupby("Type 1")
print((group['HP'].mean().to_string()))
print((group['HP'].sum().to_string()))
print((group['HP'].min().to_string()))
print((group['HP'].max().to_string()))
print((group['HP'].count().to_string()))
