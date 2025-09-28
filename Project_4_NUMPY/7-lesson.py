import numpy as np

# Filtering = Refers to the processof selecting elements from an array that match a given condition

ages = np.array([[21,17,19,20,16,30,18,65],
                 [39,22,15,99,18,19,20,21]])
'''
teens = ages[ages < 18]
adults = ages[(ages >= 18) & (ages < 65)]
seniors = ages[ages >= 65]
evens = ages[ages % 2 == 0]
odds = ages[ages % 2 != 0]

print(teens)
print(adults)
print(seniors)
print(evens)
print(odds)
'''

adults = np.where(ages >= 18, ages, 0)
print(adults)