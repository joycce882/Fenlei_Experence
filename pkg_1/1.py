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
                # 计算后验概率的对数
                class_prob = np.log(self.class_prob[i])
                feature_probs = np.log(self.feature_prob[i])
                # 计算条件概率的对数
                class_probs.append(class_prob + np.sum(feature_probs * x))
            # 选择后验概率最大的类别作为预测结果
            y_pred.append(self.classes[np.argmax(class_probs)])
        return y_pred


# 读取鸢尾花数据集
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                   header=None, names=["sepal length", "sepal width", "petal length", "petal width", "class"])
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 训练模型
nb = NaiveBayes()
nb.fit(X, y)


# 预测一条数据
test_data = np.array([[10.5,3.5,15.1,1.7]])
y_pred = nb.predict(test_data)
print("Predicted class:", y_pred[0])
