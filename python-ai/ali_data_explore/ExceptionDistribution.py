import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge


# 数据的异常值分布
# https://www.debugger.wiki/article/html/1563068783112436 岭回归
def find_outliners(model, X, y, sigma=3):
    # 使用模型预测值
    # try:
    #    ridge_model =  y_pred = pd.Series(model.predict(X), index=y.index)
    #
    # # 如果预测失败，首先尝试拟合模型
    # except :
    model.fit(X, y)
    y_pred = pd.Series(model.predict(X), index=y.index)

    # 计算模型预测和真实y值之间的残差
    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()

    # 计算z统计量，定义离群点在|z|>处
    z = (resid - mean_resid) / std_resid
    outliners = z[abs(z) > sigma].index

    # 打印并绘制结果
    print("R2=", model.score(X, y))
    print("mse=", mean_squared_error(y, y_pred))
    print('---------------------------------------')

    print("mean of residuals", mean_resid)
    print("std of residuals", std_resid)
    print('---------------------------------------')

    print(len(outliners), "outliers:")
    print(outliners.tolist())

    plt.figure(figsize=(15, 5))
    ax_131 = plt.subplot(1, 3, 1)
    plt.plot(y, y_pred, ".")
    plt.plot(y.loc[outliners], y_pred.loc[outliners], 'ro')
    plt.legend(['Accepted', 'Outlier'])
    plt.xlabel('y')
    plt.ylabel('y_pred');

    ax_132 = plt.subplot(1, 3, 2)
    plt.plot(y, y - y_pred, '.')
    plt.plot(y.loc[outliners], y.loc[outliners] - y_pred.loc[outliners], 'ro')
    plt.legend(['Accepted', 'Outlier'])
    plt.xlabel('y')
    plt.ylabel('y - y_pred');

    ax_133 = plt.subplot(1, 3, 3)
    z.plot.hist(bins=50, ax=ax_133)
    z.loc[outliners].plot.hist(color='r', bins=50, ax=ax_133)
    plt.legend(['Accepted', 'Outlier'])
    plt.xlabel('z')
    plt.show()
    # plt.savefig('outliers.png')

    return outliners


if __name__ == '__main__':
    train_data_file = "../data/zhengqi_train.txt"
    test_data_file = "../data/zhengqi_test.txt"

    train_data = pd.read_csv(train_data_file, sep='\t', encoding='utf-8')
    test_data = pd.read_csv(test_data_file, sep='\t', encoding='utf-8')

    # print(train_data)
    print("---------------")
    x_train = train_data.iloc[:, 0:-1]
    y_train = train_data.iloc[:, -1]
    # print(x_train)
    outliers = find_outliners(Ridge(), x_train, y_train)
