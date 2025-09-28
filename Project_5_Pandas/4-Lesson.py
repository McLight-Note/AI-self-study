import pandas as pd

df = pd.read_csv("pokemon.csv")

# Selection by column
print(df[['Name', 'HP', "Speed"]].to_string())

# Selection by row
print(df.iloc[1])

# Which index you like first
df_col = pd.read_csv('pokemon.csv', index_col="Name")
print(df_col)
print(df_col.loc['Pikachu'])
print(df_col.loc['Charizard' : 'Blastoise', ['HP', 'Speed']])

print(df_col.iloc[:11:2, 0:3])

pokemon = input('Enter a Pokemon name: ')

try:
    print(df_col.loc[pokemon])
except KeyError:
    print(f"That {pokemon} not found")