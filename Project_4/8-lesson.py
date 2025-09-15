import numpy as np

'''rng = np.random.default_rng(seed=1)

print(rng.integers(low=1, high=101, size=(3,2)))

# np.random(seed=1)

print(np.random.uniform(low=-1, high=1, size=(3,2)))'''

rng = np.random.default_rng()

array = np.array([1,2,3,4,5])

# rng.shuffle(array)

print(array)

fruits = np.array(['apple', 'orange', 'banana', 'coconut', 'pinapple'])

fruits = rng.choice(fruits, size=(4,2))

print(fruits)