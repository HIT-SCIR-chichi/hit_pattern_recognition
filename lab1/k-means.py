"""
作者：张景润
学号：1172510217
"""
import csv
import random
import numpy as np

Num = 10  # 聚类的类别数目
Width = 784  # 样本的特征维度
Data_Path = './ClusterSamples.csv'  # 样本所在的文件
Ground_Path = './SampleLabels.csv'  # 真实聚类结果文件
Label_Path = './ClusterLabels.csv'  # 样本聚类标签输出文件


def read_data():
    global Width
    with open(Data_Path, 'r') as f:
        data_lst = []
        reader = csv.reader(f)
        for lst in reader:
            data_lst.append([int(item) for item in lst])
        Width = len(data_lst[0])
        return data_lst


def init_cluster_center(data_lst, init=1):  # 初始化聚类中心
    if init == 0:  # 随机初始化聚类中心
        center_lst = random.sample(data_lst, Num)
    elif init == 1:  # 选取批次距离尽可能远的Num个点
        center_lst = [get_center(data_lst)]  # 第一个样本点为所有样本点的质心
        while len(center_lst) < Num:
            dis, new_center = 0, []
            for data in data_lst:
                new_dis = min([cal_dis(data, center) for center in center_lst])
                if new_dis >= dis:
                    dis = new_dis
                    new_center = data
            center_lst.append(new_center)  # 选取距离已有中心点的最近距离最大的点作为新的中心点
    else:  # 待开发
        center_lst = init_cluster_center(data_lst, 0)
    return center_lst


def cal_dis(data1, data2):  # 计算两个样本的中心
    return sum([(item1 - item2) ** 2 for item1, item2 in zip(data1, data2)])


def get_center(data_lst):  # 获取data_lst的中心
    center = [0] * Width
    for data in data_lst:
        for idx, item in enumerate(data):
            center[idx] += item
    center = [float(item) / len(data_lst) for item in center]
    return center


def cluster(data_lst, center_lst):
    label_lst = [-1] * len(data_lst)
    flag, iter_count = True, 0
    while flag:
        flag = False
        for idx, data in enumerate(data_lst):  # 更新每一个聚类的标签
            dis_lst = [cal_dis(data, data2) for data2 in center_lst]
            min_idx = dis_lst.index(min(dis_lst))  # 找到与当前样本距离最近的聚类中心
            if min_idx != label_lst[idx]:
                label_lst[idx] = min_idx
                flag = True
        for idx in range(Num):  # 更新聚类中心
            tmp_lst = list(map(lambda item: item[1], filter(lambda item: item[0] == idx, zip(label_lst, data_lst))))
            center_lst[idx] = get_center(tmp_lst)
        iter_count += 1
        print('当前是第%d次迭代' % iter_count)
    return label_lst


def write_data(label_lst):  # 将聚类结果写入文件
    with open(Label_Path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(map(str, label_lst)))


def evaluate():  # 查看聚类结果与标准答案比较
    with open(Ground_Path, 'r', encoding='utf-8') as f0, open(Label_Path, 'r', encoding='utf-8') as f1:
        ground_lst, predict_lst = [], []
        cluster_lst = [[0 for _ in range(Num)] for _ in range(Num)]
        for line in f0:
            line = line.strip('\n')
            if line:
                ground_lst.append(int(line))
        for line in f1:
            line = line.strip('\n')
            if line:
                predict_lst.append(int(line))
        for ground, predict in zip(ground_lst, predict_lst):
            cluster_lst[ground][predict] += 1
        for idx, lst in enumerate(cluster_lst):
            print('聚类%d：%s' % (idx, '\t'.join(map(str, lst))))
        res = np.array(cluster_lst)  # debug模式下以表格形式查看res
        print(res)


def main():
    data_lst = read_data()
    center_lst = init_cluster_center(data_lst)
    label_lst = cluster(data_lst, center_lst)
    write_data(label_lst)
    # evaluate()


if __name__ == '__main__':
    main()
