import numpy as np
import pandas as pd


class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_prob = None
        self.feature_prob = None

    def fit(self, X, y):
        # 获取类别
        self.classes = np.unique(y)
        num_classes = len(self.classes)
        num_features = X.shape[1]
        # 初始化先验概率和条件概率
        self.class_prob = np.zeros(num_classes)
        self.feature_prob = np.zeros((num_classes, num_features,2))

        # 计算先验概率和条件概率
        for i, c in enumerate(self.classes):
            # 获取属于当前类别的样本
            X_c = X[c == y]
            # 计算先验概率
            self.class_prob[i] = len(X_c) / len(X)
            # 计算条件概率
            for j in range(num_features):
                # 属于当前类别中第 j 个特征的所有值
                values = X_c[:, j]
                # 计算第 j 个特征的均值和标准差
                mean, std = np.mean(values), np.std(values)
                # 存储第 j 个特征的均值和标准差
                self.feature_prob[i, j] = [mean, std]

    def predict(self, X):
        y_pred = []
        for x in X:
            class_probs = []
            for i, c in enumerate(self.classes):
                # 计算后验概率的对数
                class_prob = np.log(self.class_prob[i])
                feature_probs = 0
                for j in range(len(x)):
                    # 获取第 j 个特征的均值和标准差
                    mean, std = self.feature_prob[i, j]
                    # 计算第 j 个特征下给定样本 x 的概率密度函数的值
                    p = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-(x[j] - mean) ** 2 / (2 * std ** 2))
                    # 累乘每个特征的条件概率
                    feature_probs += np.log(p)
                # # 计算该样本属于类别 c 的后验概率
                class_probs.append(class_prob + feature_probs)
            # 选择后验概率最大的类别作为预测结果
            y_pred.append(self.classes[np.argmax(class_probs)])
        return y_pred

# 读取鸢尾花数据集
data = pd.read_csv("iris.csv", header=None,
                   names=["sepal length", "sepal width", "petal length", "petal width", "class"])
data1 = pd.DataFrame(data)
data1 = data1.values
data = np.delete(data1, 0, axis=0)
# data = np.delete(data, 0, axis=1)
np.random.shuffle(data)
test_redio = 0.3
test_size = int(len(data) * test_redio)
X_train = data[:len(data) - test_size, :-1]
X_test = data[len(data) - test_size:, :-1]
# X = data[0:, :-1]
y_train = data[:len(data) - test_size, -1]
y_test = data[len(data) - test_size:, -1]
X_1 = X_train.astype(float)
X_test = X_test.astype(float)
print(type(y_test))
# print(X_1)


# print(X_1)
# X_2 = np.zeros((4,), dtype='i,f')
# print(X_2)


# 训练模型
nb = NaiveBayes()
nb.fit(X_1, y_train)

j = 0
# 预测多条数据

y_pred = nb.predict(X_test)


for i in range(len(y_pred)):
    print("判断种类分别为："+str(i)+",:"+y_pred[i])
    if y_pred[i] == y_test[i]:
        j += 1
m = X_test.shape[0]
print("正确率为：" + str(j / m))


