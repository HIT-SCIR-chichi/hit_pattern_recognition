"""
作者：张景润
学号：1172510217
"""
import argparse
import csv
import numpy as np


def read_data(path):  # 读取源数据csv文件
    with open(path, 'r') as f:
        data_lst = []
        reader = csv.reader(f)
        for lst in reader:
            data_lst.append(list(map(float, lst)))
        return np.array(data_lst)


def read_mnist_data(train=True):
    data_path = './TrainSamples.csv' if train else './TestSamples.csv'
    label_path = './TrainLabels.csv' if train else './TestLabels.csv'

    data_np, label_np = read_data(data_path), read_data(label_path)
    if train:
        data_num = label_np.shape[0]
        split_data = [list() for _ in range(10)]
        for i in range(data_num):
            split_data[int(label_np[i])].append(data_np[i])
        priors = [len(split_data[i]) / data_num for i in range(10)]
        split_data = [np.array(lst) for lst in split_data]
        return split_data, priors
    else:
        return data_np, label_np


class GMM:
    def __init__(self, gauss_num):
        self.gauss_num = gauss_num  # 高斯混合分布的数目

        self.alphas_np = None  # 每一个高斯分布的权重
        self.means_np = None  # 各维变量的均值
        self.covs_np = None  # 协方差矩阵

    def init_params(self, data_np):  # 初始化高斯分布的三部分参数
        data_shuffled = data_np.copy()
        np.random.shuffle(data_shuffled)
        data_split = np.array_split(data_shuffled, self.gauss_num)

        self.alphas_np = np.repeat(1.0 / self.gauss_num, self.gauss_num)
        self.means_np = np.array([np.mean(data_split[i], axis=0) for i in range(self.gauss_num)])
        self.covs_np = np.array([np.cov(data_split[i].T) for i in range(self.gauss_num)])

    def pdf(self, x_np, idx):  # 概率密度计算
        mean_np = self.means_np[idx]
        cov_np = self.covs_np[idx]
        left = 1 / np.sqrt(np.power(2 * np.pi, x_np.shape[0]) * np.linalg.det(cov_np))
        right = np.exp(-0.5 * np.dot(np.dot((x_np - mean_np).T, np.linalg.inv(cov_np)), x_np - mean_np))
        return left * right

    def fit(self, data_np, fit_num, eps):  # EM过程计算参数
        self.init_params(data_np)

        data_num = data_np.shape[0]  # 数据数目
        norm_densities = np.empty((data_num, self.gauss_num), np.float)
        responsibilities = np.empty((data_num, self.gauss_num), np.float)
        pre_log_likelihood = 0
        for idx in range(fit_num):
            for i, one_data in enumerate(data_np):
                for j in range(self.gauss_num):
                    norm_densities[i][j] = self.pdf(one_data, j)
            log_vec = np.log(np.array([np.dot(self.alphas_np, norm_density) for norm_density in norm_densities]))
            log_likelihood = log_vec.sum()
            if abs(log_likelihood - pre_log_likelihood) < eps:
                break

            for i in range(data_num):
                normalizer = np.dot(self.alphas_np.T, norm_densities[i])
                for j in range(self.gauss_num):
                    responsibilities[i][j] = self.alphas_np[j] * norm_densities[i][j] / normalizer

            for i in range(self.gauss_num):
                responsibility = responsibilities.T[i]
                normalizer = np.dot(responsibility, np.ones(data_num))
                self.alphas_np[i] = normalizer / data_num
                self.means_np[i] = np.dot(responsibility, data_np) / normalizer
                diff = data_np - np.tile(self.means_np[i], (data_num, 1))
                self.covs_np[i] = np.dot((responsibility.reshape(data_num, 1) * diff).T, diff) / normalizer

            pre_log_likelihood = log_likelihood
            print('第%d次迭代' % (idx + 1))

        print(self.alphas_np)
        print(self.means_np)
        print(self.covs_np)


class Classifier:
    def __init__(self, gmm_lst, weight_lst):
        self.gmm_lst = gmm_lst
        self.weight_lst = weight_lst
        self.classes = len(weight_lst)

    def classify(self, data_np, label):
        data_num = data_np.shape[0]
        if isinstance(label, int):
            label = np.full((data_num,), label)
        log_vec = np.empty((self.classes, data_num), dtype=np.float)
        for idx, gmm in enumerate(self.gmm_lst):
            pdf_np = np.array([[gmm.pdf(data_np[i], j) for j in range(gmm.gauss_num)] for i in range(data_num)])
            log_vec[idx] = np.array([np.dot(gmm.alphas_np, pdf_lst) for pdf_lst in pdf_np]) * self.weight_lst[idx]

        predict_np = np.argmax(log_vec, axis=0)
        right_num = (predict_np == label.reshape(predict_np.shape)).sum()
        accuracy = right_num / data_num
        print('[%d/%d]=%.2f%%' % (right_num, data_num, accuracy * 100))


def main():  # 训练单个gmm模型
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./Train1.csv', help='训练文件路径，默认为./Train1.csv')
    parser.add_argument('--gauss_num', type=int, default=2, help='高斯混合分布的数目，默认为2')
    parser.add_argument('--fit_num', type=int, default=20, help='EM步骤的最大迭代次数，默认为20')
    parser.add_argument('--eps', type=float, default=1e-6, help='EM步骤的最小对数似然估计loss差值，默认为1e-6')
    args = parser.parse_args()
    data_path, gauss_num, fit_num, eps = args.data_path, args.gauss_num, args.fit_num, args.eps

    data_np = read_data(data_path)
    gmm = GMM(gauss_num)
    gmm.fit(data_np, fit_num, eps)


def classify():  # 两个gmm模型混合，并对结果进行分类
    gmm1 = GMM(2)
    gmm2 = GMM(2)
    gmm1.fit(read_data('./Train1.csv'), 20, 1e-6)
    gmm2.fit(read_data('./Train2.csv'), 20, 1e-6)

    classifier = Classifier([gmm1, gmm2], [0.5, 0.5])
    classifier.classify(read_data('./Test1.csv'), 0)
    classifier.classify(read_data('./Test2.csv'), 1)


def classify_mnist():  # 训练多个gmm混合模型，并对结果进行分类
    parser = argparse.ArgumentParser()
    parser.add_argument('--mixture_num', type=int, default=1, help='高斯混合分布数目，默认为1')
    args = parser.parse_args()
    mixture_num = args.mixture_num

    print('K=%d' % mixture_num)
    gmm_lst = [GMM(mixture_num) for _ in range(10)]
    train_data, priors = read_mnist_data(train=True)
    for idx, (gmm, data) in enumerate(zip(gmm_lst, train_data)):
        print('当前训练的分类器为：%d' % (idx + 1))
        gmm.fit(data, 40, 1e-10)

    mnist_classifier = Classifier(gmm_lst, priors)
    test_data, test_label = read_mnist_data(train=False)
    mnist_classifier.classify(test_data, test_label)


if __name__ == '__main__':
    # main()
    # classify()
    classify_mnist()
