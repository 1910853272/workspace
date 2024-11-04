import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import matplotlib.pyplot as plt

# 第一步：加载 MNIST 数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 将数据转换为 numpy 数组
X_train = mnist_train.data.numpy().reshape(-1, 28*28)
y_train = mnist_train.targets.numpy()
X_test = mnist_test.data.numpy().reshape(-1, 28*28)
y_test = mnist_test.targets.numpy()

# 第二步：使用 PCA 和 LDA 进行降维
def dimensionality_reduction(X_train, X_test, y_train, method, n_components):
    if method == 'PCA':
        reducer = PCA(n_components=n_components)
        X_train_reduced = reducer.fit_transform(X_train)
        X_test_reduced = reducer.transform(X_test)
    elif method == 'LDA':
        # LDA 的 n_components 不能超过 min(n_features, n_classes - 1)
        n_components = min(n_components, min(X_train.shape[1], len(np.unique(y_train)) - 1))
        reducer = LDA(n_components=n_components)
        X_train_reduced = reducer.fit_transform(X_train, y_train)
        X_test_reduced = reducer.transform(X_test)
    else:
        raise ValueError("Method should be either 'PCA' or 'LDA'")
    return X_train_reduced, X_test_reduced

# 第三步：训练和评估 QDF 分类器
# 使用正则化判别分析（RDA）的二次判别函数（QDF）
def train_qdf_with_rda(X_train, y_train, X_test, y_test, reg_param):
    qda = QDA(reg_param=reg_param)
    qda.fit(X_train, y_train)
    y_pred = qda.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# 不使用 RDA 的 QDF 分类器
def train_qdf_without_rda(X_train, y_train, X_test, y_test):
    qda = QDA()  # 没有设置正则化参数
    qda.fit(X_train, y_train)
    y_pred = qda.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# 第四步：在不同的降维子空间中比较使用 RDA 和不使用 RDA 的 QDF 分类器
dimensions = [2, 5, 10, 20, 50]
methods = ['PCA', 'LDA']
regularization_params = [0.1, 0.5, 0.9]

results_with_rda = {}
results_without_rda = {}

for method in methods:
    results_with_rda[method] = {}
    results_without_rda[method] = {}
    for n_components in dimensions:
        results_with_rda[method][n_components] = []
        results_without_rda[method][n_components] = []
        X_train_reduced, X_test_reduced = dimensionality_reduction(X_train, X_test, y_train, method, n_components)

        # 使用 RDA 的 QDF 分类器
        for reg_param in regularization_params:
            acc_with_rda = train_qdf_with_rda(X_train_reduced, y_train, X_test_reduced, y_test, reg_param)
            results_with_rda[method][n_components].append(acc_with_rda)

        # 不使用 RDA 的 QDF 分类器
        acc_without_rda = train_qdf_without_rda(X_train_reduced, y_train, X_test_reduced, y_test)
        results_without_rda[method][n_components] = acc_without_rda

# 第五步：结果的可视化
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
for idx, method in enumerate(methods):
    for reg_idx, reg_param in enumerate(regularization_params):
        accuracies_with_rda = [results_with_rda[method][n][reg_idx] for n in dimensions]
        ax[idx].plot(dimensions, accuracies_with_rda, label=f'With RDA (Reg Param: {reg_param})')

    accuracies_without_rda = [results_without_rda[method][n] for n in dimensions]
    ax[idx].plot(dimensions, accuracies_without_rda, label='Without RDA', linestyle='--', color='black')

    ax[idx].set_title(f'QDF Accuracy Comparison with {method}')
    ax[idx].set_xlabel('Number of Components')
    ax[idx].set_ylabel('Accuracy')
    ax[idx].legend()

plt.tight_layout()
plt.savefig('./result/qdf_comparison_with_without_rda.png')
plt.show()
