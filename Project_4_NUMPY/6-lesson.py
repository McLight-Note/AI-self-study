import numpy as np
# aggregate funcs = summarize data and typically return a single value
array = np.array([[1,2,3,4,5],
                  [6,7,8,9,10]])
'''
print(np.sum(array))
print(np.mean(array))
print(np.std(array)) # Stan deviation
print(np.var(array))
print(np.min(array))
print(np.max(array))
print(np.argmin(array))
print(np.argmax(array))
'''

print(np.sum(array, axis=1))