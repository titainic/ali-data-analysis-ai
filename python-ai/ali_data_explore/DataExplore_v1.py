import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

train_data_file = "../data/zhengqi_train.txt"
test_data_file = "../data/zhengqi_test.txt"

train_data = pd.read_csv(train_data_file, sep="\t", encoding="utf-8")
test_data = pd.read_csv(test_data_file, sep="\t", encoding="utf-8")

# train_data.info()
# test_data.info()

# 统计信息，均值。最大，最小，标准差
print(train_data.describe())

# 指定绘图对象的宽度和高度
fig = plt.figure(figsize=(4, 6))
sns.boxplot(train_data['V0'], orient='v', width=0.5)
plt.show()

column = train_data.columns.tolist()
fig = plt.figure(figsize=(80, 60), dpi=75)
for i in range(38):
    plt.subplot(7, 6, i + 1)
    sns.boxplot((train_data[column[i]]),orient="v",width=0.5)
    plt.ylabel(column[i],fontsize=35)
plt.show()






