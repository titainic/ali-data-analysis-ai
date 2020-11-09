import numpy as np

#相关性。正相关，负相关，不相关
X = np.array([65, 72, 78, 65, 72, 70, 65, 68])
Y = np.array([72, 69, 79, 69, 84, 75, 60, 73])

print(np.corrcoef(X, Y))
