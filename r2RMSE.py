#coding=utf-8
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def get_average(records):
    """
    平均值
    """
    return sum(records) / len(records)


def get_variance(records):
    """
    方差 反映一个数据集的离散程度
    """
    average = get_average(records)
    return sum([(x - average) ** 2 for x in records]) / len(records)


def get_standard_deviation(records):
    """
    标准差 == 均方差 反映一个数据集的离散程度
    """
    variance = get_variance(records)
    return math.sqrt(variance)


def get_rms(records):
    """
    均方根值 反映的是有效值而不是平均值
    """
    return math.sqrt(sum([x ** 2 for x in records]) / len(records))


def get_mse(records_real, records_predict):
    """
    均方误差 估计值与真值 偏差
    """
    if len(records_real) == len(records_predict):
        return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None


def get_rmse(records_real, records_predict):
    """
    均方根误差：是均方误差的算术平方根
    """
    mse = get_mse(records_real, records_predict)
    if mse:
        return math.sqrt(mse)
    else:
        return None


def get_mae(records_real, records_predict):
    """
    平均绝对误差
    """
    if len(records_real) == len(records_predict):
        return sum([abs(x - y) for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None

def pearson(df):
    print(df)
    a=df.corr()
    print('皮尔逊系数')
    print(a)
    #绘制热图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    sns.heatmap(df.corr(method='pearson'),linewidths=0.1,vmax=1.0, square=True,linecolor='white', annot=True)
    plt.title('Pearson heatmap')
    plt.show()

if __name__ == '__main__':
    records1 = [3, 4, 5]
    records2 = [2, 4, 6]

    # 平均值
    average1 = get_average(records1)  # 4.0
    average2 = get_average(records2)  # 4.0

    # 方差
    variance1 = get_variance(records1)  # 0.66
    variance2 = get_variance(records2)  # 2.66

    # 标准差
    std_deviation1 = get_standard_deviation(records1)  # 0.81
    std_deviation2 = get_standard_deviation(records2)  # 1.63

    # 均方根
    rms1 = get_rms(records1)  # 4.08
    rms2 = get_rms(records2)  # 4.32

    # 均方误差
    mse = get_mse(records1, records2)  # 0.66

    # 均方根误差
    rmse = get_rmse(records1, records2)  # 0.81

    # 平均绝对误差
    mae = get_mae(records1, records2)  # 0.66

    dates = pd.date_range('20220101', periods=15)#生成连续六天的时间序列
    #随机生成数据
    df = pd.DataFrame(np.random.randn(15,4), index=dates, columns=list('ABCD'))
    pearson(df)