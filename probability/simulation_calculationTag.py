# -*- coding: utf-8 -*-
# Author: lzjiang
import numpy as np
import matplotlib.pyplot as plt

data = np.random.rand(1, 100)
print(data.shape)

convert = np.random.weibull(0.7, (1, 100))
x = np.asarray([1, 100])
print(convert)

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.scatter(data, convert)
plt.show()
