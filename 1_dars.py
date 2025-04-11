# 2025.04.04
# Mavzu: Intro to AI
# Muallif: Muhammadsodiq

import numpy
import matplotlib.pyplot

a = numpy.zeros([3,2])
a[0,0] = 1
a[0,1] = 4
a[1,0] = 3
a[1,1] = 3
a[2,0] = 5
a[2,1] = 4
print(a)   

matplotlib.pyplot.imshow(a, interpolation='nearest')
matplotlib.pyplot.show()