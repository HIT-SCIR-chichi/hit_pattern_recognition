"""
作者：张景润
学号：1172510217
"""
import numpy as np


def get_data():
    return np.array([[1, 1], [2, 2], [2, 0], [0, 0], [1, 0], [0, 1]]), np.array([0, 0, 0, 1, 1, 1])


def read_data(path, dtype=float):
    return np.genfromtxt(path, delimiter=',', dtype=dtype)


def data_augment(data_np, label_np):  # 增广数据，将数值1添加到第0列，并将标签为1的数据取反
    data_np = np.insert(data_np, 0, 1, axis=1)
    for idx, label in enumerate(label_np):
        if label:
            data_np[idx] = - data_np[idx]
    return data_np


def predict(w_np, data_np, label_np):
    data_np, data_num = np.insert(data_np, 0, 1, axis=1), len(data_np)
    predict_np = np.array([0 if np.dot(w_np, data) > 0 else 1 for data in data_np])
    right_num = int(np.sum(predict_np == label_np))
    print('[%d/%d]=%.2f%%' % (right_num, data_num, (right_num / data_num) * 100))


def perceptron_fit(data_np, label_np):  # 感知器二分类
    data_np = data_augment(data_np, label_np)
    data_num, data_width = data_np.shape
    k, w_np = 0, np.random.randn(data_width)
    while True:
        if np.dot(w_np, data_np[k]) <= 0:
            w_np = w_np + data_np[k]
        k = (k + 1) % data_num
        flag = np.sum([np.dot(w_np, data) <= 0 for data in data_np])  # 查看错误分类的数目
        if not flag:
            break
    return w_np


def lmse_fit(data_np, label_np):  # 最小平方误差准则进行二分类Least Minimum Squared Error
    data_np = data_augment(data_np, label_np)
    data_num, data_width = data_np.shape
    label_np = np.ones((data_num, 1))
    w_np = np.linalg.inv(data_np.T.dot(data_np)).dot(data_np.T).dot(label_np).reshape((data_width,))
    return w_np


def predict_multi(w_np_lst, c, data_np, label_np, fit=False):  # 多分类中的预测函数
    if not fit:
        data_np = np.insert(data_np, 0, 1, axis=1)
    data_num = len(data_np)
    predict_np = np.array([np.argmax([np.dot(w_np_lst[i], data) for i in range(c)]) for data in data_np])
    right_num = (predict_np == label_np).sum()
    print('[%d/%d]=%.2f%%' % (right_num, data_num, (right_num / data_num) * 100))
    return right_num == data_num


class KeslerPerceptron:
    def __init__(self, c=10):
        self.c = c
        self.w_np_lst = []

    def fit(self, data_np, label_np, iter_num=100, lr=1e-7):
        data_np = np.insert(data_np, 0, 1, axis=1)
        data_num, data_width = data_np.shape

        k, self.w_np_lst = 0, [np.random.randn(data_width) for _ in range(self.c)]
        iter_count = 0
        while True:
            data, label = data_np[k], label_np[k]
            g_label = np.dot(self.w_np_lst[label], data)
            for idx in range(self.c):
                g_idx = np.dot(self.w_np_lst[idx], data)
                if idx != label and g_idx >= g_label:
                    self.w_np_lst[label] += lr * data
                    self.w_np_lst[idx] -= lr * data
            k = (k + 1) % data_num
            if k == 0:
                iter_count += 1
                print('当前迭代次数为：%d' % iter_count)
                if iter_count == iter_num or predict_multi(self.w_np_lst, self.c, data_np, label_np, fit=True):
                    break
        print('训练集上得到的模型最终在训练集上的分类效果如下：')
        predict_multi(self.w_np_lst, self.c, data_np, label_np, fit=True)


class LmseOva:
    def __init__(self, c=10):
        self.c = c
        self.w_np_lst = []

    def fit(self, data_np, label_np):
        def get_ova_label(data, label, cls):
            label1 = label.copy()
            label1[label == cls], label1[label != cls] = 0, 1
            return data, label1

        for idx in range(self.c):
            data_np, label_new = get_ova_label(data_np, label_np, idx)
            self.w_np_lst.append(lmse_fit(data_np, label_new))
        print('训练集上得到的模型最终在训练集上的分类效果如下：')
        predict_multi(self.w_np_lst, self.c, data_np, label_np, fit=False)


def perceptron_main():
    data_np, label_np = get_data()
    w_np = perceptron_fit(data_np, label_np)
    predict(w_np, data_np, label_np)
    print(w_np)


def lmse_main():
    data_np, label_np = get_data()
    w_np = lmse_fit(data_np, label_np)
    predict(w_np, data_np, label_np)
    print(w_np)


def kesler_perceptron_main():
    kesler = KeslerPerceptron(c=10)
    kesler.fit(read_data('./TrainSamples.csv'), read_data('./TrainLabels.csv', dtype=int))
    print('模型最终在测试集上的分类效果如下：')
    predict_multi(kesler.w_np_lst, kesler.c, read_data('./TestSamples.csv'), read_data('./TestLabels.csv', dtype=int))


def lmse_ova_main():
    ova = LmseOva(10)
    ova.fit(read_data('./TrainSamples.csv'), read_data('./TrainLabels.csv', dtype=int))
    print('模型最终在测试集上的分类效果如下：')
    predict_multi(ova.w_np_lst, ova.c, read_data('./TestSamples.csv'), read_data('./TestLabels.csv', dtype=int))


if __name__ == '__main__':
    # perceptron_main()
    # lmse_main()
    kesler_perceptron_main()
    # lmse_ova_main()
