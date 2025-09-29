import pandas as pd

df = pd.read_csv('pokemon.csv')

healthy_pokemon = df[df['HP'] >= 200]
print(healthy_pokemon)

fast_pokemon = df[df['Speed'] >= 100]
print(fast_pokemon)

legend_pokemon = df[df['Legendary']]
print(legend_pokemon)

water_pokemon = df[(df['Type 1'] == 'Water') |
                   (df['Type 2'] == 'Water')]
print(water_pokemon)

ff_pokemon = df[(df['Type 1'] == "Fire") &
                (df['Type 2'] == "Flying")]
print(ff_pokemon)

# Homework

nan_new = df.isna().sum()
print(nan_new)