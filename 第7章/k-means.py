# coding=utf-8
# author:yjy
# date:2019/12/1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time

# 西瓜数据集4.0: 密度 含糖率 标签
data = [[0.697, 0.460, 1],
        [0.774, 0.376, 1],
        [0.634, 0.264, 1],
        [0.608, 0.318, 1],
        [0.556, 0.215, 1],
        [0.430, 0.237, 1],
        [0.481, 0.149, 1],
        [0.437, 0.211, 1],
        [0.666, 0.091, 0],
        [0.243, 0.267, 0],
        [0.245, 0.057, 0],
        [0.343, 0.099, 0],
        [0.639, 0.161, 0],
        [0.657, 0.198, 0],
        [0.360, 0.370, 0],
        [0.593, 0.042, 0],
        [0.719, 0.103, 0],
        [0.359, 0.188, 0],
        [0.339, 0.241, 0],
        [0.282, 0.257, 0],
        [0.748, 0.232, 0],
        [0.714, 0.346, 1],
        [0.483, 0.312, 1],
        [0.478, 0.437, 1],
        [0.525, 0.369, 1],
        [0.751, 0.489, 1],
        [0.532, 0.472, 1],
        [0.473, 0.376, 1],
        [0.725, 0.445, 1],
        [0.446, 0.459, 1]]

column = ['density', 'sugar_rate', 'label']
dataSet = pd.DataFrame(data, columns=column)


class K_means(object):
    def __init__(self, k, data, loop_times, error):  # self只有在类的方法中才会有，指向类的实例对象，而非类本身
        self.k = k
        self.data = data
        self.loop_times = loop_times
        self.error = error

    def distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def fitting(self):
        time1 = time.perf_counter()
        mean_vectors = random.sample(self.data, self.k)  # 随机选取k个初始样本
        initial_main_vectors = mean_vectors
        for vec in mean_vectors :
            plt.scatter(vec[0], vec[1], s=100, color = 'black', marker='s')
        times = 0
        clusters = list(map((lambda x:[x]), mean_vectors))
        while times < self.loop_times:
            change_flag = 1  # 标记簇均值向量是否改变
            for sample in self.data:
                dist = []
                for vec in mean_vectors:
                    dist.append(self.distance(vec, sample))  # 计算样本到每个聚类中心的距离
                clusters[dist.index(min(dist))].append(sample)  # 找到离该样本最近的聚类中心，并将它放入该簇

            new_mean_vectors = []
            for c,v in zip(clusters, mean_vectors):  # zip()将两个对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
                cluster_num = len(c)
                cluster_array = np.array(c)
                new_mean_vector = sum(cluster_array) / cluster_num  # 计算出新的聚类簇均值向量
                mean_vector = np.array(v)
                if all(np.true_divide((new_mean_vector - mean_vector), mean_vector) < np.array([self.error, self.error])):
                    new_mean_vectors.append(mean_vector)  # 均值向量未改变
                    change_flag = 0
                else:
                    # dataFrame转List()，括号不能忘
                    new_mean_vectors.append(new_mean_vector.tolist())  # 均值向量发生改变

            if change_flag == 1:
                mean_vectors = new_mean_vectors
            else:
                break
            times += 1
        time2 = time.perf_counter()
        # str.format()，基本语法是通过 {} 和 : 来代替以前的 %
        print ('本次选取的{}个初始向量为{}'.format(self.k, initial_main_vectors))
        print ('共进行{}轮'.format(times))
        print ('共耗时{:.2f}s'.format(time2 - time1))  # 取2位小数
        for cluster in clusters:
            x = list(map(lambda arr: arr[0], cluster))
            y = list(map(lambda arr: arr[1], cluster))
            plt.scatter(x, y, marker = 'o', label = clusters.index(cluster)+1)
        plt.xlabel('密度')
        plt.ylabel('含糖率')
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.legend(loc='upper left')
        plt.show()


# 调用K_means，执行方法fitting()
k_means = K_means(j, dataSet[['density', 'sugar_rate']].values.tolist(), 1000, 0.0000001)
k_means.fitting()