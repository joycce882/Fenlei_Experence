import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


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
        self.feature_prob = np.zeros((num_classes, num_features))

        # 计算先验概率和条件概率
        for i, c in enumerate(self.classes):
            # 获取属于当前类别的样本
            X_c = X[c == y]
            # 计算先验概率
            self.class_prob[i] = len(X_c) / len(X)
            # 计算条件概率
            self.feature_prob[i] = np.mean(X_c, axis=0)

    def predict(self, X):
        y_pred = []
        for x in X:
            class_probs = []
            for i, c in enumerate(self.classes):
                # 计算先验概率的对数
                class_prob = np.log(self.class_prob[i])
                # 计算条件概率的对数
                feature_probs = np.log(self.feature_prob[i])

                class_probs.append(class_prob + np.sum(feature_probs * x))
            # 选择先验概率最大的类别作为预测结果
            y_pred.append(self.classes[np.argmax(class_probs)])
        return y_pred


# 读取鸢尾花数据集
data = pd.read_csv("iris.csv", header=None, names=["sepal length", "sepal width", "petal length", "petal width", "class"])
data1 = pd.DataFrame(data)
data1 = data1.values
data = np.delete(data1, 0, axis=0)
# data = np.delete(data, 0, axis=1)
np.random.shuffle(data)
test_redio = 0.3
test_size = int(len(data)*test_redio)
X_train = data[:len(data)-test_size, :-1]
X_test = data[len(data)-test_size:, :-1]
# X = data[0:, :-1]
y_train = data[:len(data)-test_size, -1]
y_test = data[len(data)-test_size:, -1]
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
y_pre = []
# 预测多条数据
for Evy_train in X_test:
    y_pred = nb.predict(Evy_train)
    print("Predicted class:", y_pred[0]+'\n')
    y_pre.append(y_pred[0])
for i in range(len(y_test)):
    if y_pre[i] == y_test[i]:
        j += 1
m = X_test.shape[0]
print("正确率为："+str(j / m))


