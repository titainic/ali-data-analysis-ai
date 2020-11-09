import numpy as np

'''
针对【连续、正态分布、线性】数据，采用pearson相关系数； 
针对【非线性的、非正态】数据，采用spearman相关系数； 
针对【分类变量、无序】数据，采用Kendall
相关系数。一般来讲，线性数据采用pearson，否则选择spearman，如果是分类的则用kendall。
'''
#相关性。正相关，负相关，不相关
X = np.array([65, 72, 78, 65, 72, 70, 65, 68])
Y = np.array([72, 69, 79, 69, 84, 75, 60, 73])

print(np.corrcoef(X, Y))


#pandas显示
import pandas as pd
x = [65, 72, 78, 65, 72, 70, 65, 68]
y = [72, 69, 79, 69, 84, 75, 60, 73]

df = pd.DataFrame({'x':x, 'y':y})


print(df.corr())
