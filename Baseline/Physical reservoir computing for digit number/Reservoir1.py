import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# 修改为 digit_0.npy 到 digit_9.npy，且文件路径为 'data/' 文件夹
letters = ['data/digit_0.npy', 'data/digit_1.npy', 'data/digit_2.npy', 'data/digit_3.npy',
           'data/digit_4.npy', 'data/digit_5.npy', 'data/digit_6.npy', 'data/digit_7.npy',
           'data/digit_8.npy', 'data/digit_9.npy']

# 加载 .npy 文件并存储到 ls 中
ls = list(map(np.load, letters))

# 创建字典，将文件名（如 digit_0）与相应的矩阵进行配对
d = dict(zip(list(map(lambda x: x.split('/')[-1].split('.')[0], letters)), ls))

# 打印字典中的内容，每个矩阵显示后换行
for label, matrix in d.items():
    print(f"Matrix for {label}:")
    print(matrix)
    print("\n" + "-"*40 + "\n")  # 分隔符，帮助视觉分割每个矩阵输出

# 模拟 memristor 行为的函数
def output_row(initial_state, input_signal):
    a = [initial_state]
    for i in range(len(input_signal)):  # 确保迭代次数不会超出 input_signal 的长度
        if input_signal[i] > 0:
            a.append(np.clip(a[i], 0.1, 1) * np.exp(1))
        else:
            a.append(np.clip(a[i], 1, 10) * (3 - np.exp(1)))

    # 确保返回值是 1D 数组
    return np.array(a).flatten()  # 确保 output 为一维数组

# one hot encoding, softmax function activation and training procedure
def one_hot(y, c):
    """
    # y--> label/ground truth.
    # c--> Number of classes.
    """
    # A zero matrix of size (m, c)
    y_hot = np.zeros((len(y), c))
    # Putting 1 for column where the label is,
    # Using multidimensional indexing.
    y_hot[np.arange(len(y)), y] = 1
    return y_hot

def softmax(z):
    # z--> linear part.
    # subtracting the max of z for numerical stability.
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    # Calculating softmax for all example letters.
    exp /= np.sum(exp, axis=1, keepdims=True)
    return exp

def fit(X, y, lr, c, epochs):
    """
    # X --> Input.
    # y --> true/target value.
    # lr --> Learning rate.
    # c --> Number of classes.
    # epochs --> Number of iterations.
    """
    # m-> number of training examples
    # n-> number of features
    m, n = X.shape
    # Initializing weights and bias randomly.
    w = np.random.random((n, c))
    b = np.random.random(c)
    # Empty list to store losses and accuracy.
    losses = []
    accuracies = []
    # Training loop.
    for epoch in range(epochs):
        # Calculating hypothesis/prediction.
        z = X @ w + b
        y_hat = softmax(z)
        # One-hot encoding y.
        y_hot = one_hot(y, c)
        # Calculating the gradient of loss w.r.t w and b.
        w_grad = (1 / m) * np.dot(X.T, (y_hat - y_hot))
        b_grad = (1 / m) * np.sum(y_hat - y_hot, axis=0)
        # Updating the parameters.
        w -= lr * w_grad
        b -= lr * b_grad
        # Calculating loss and appending it in the list.
        loss = -np.mean(np.log(y_hat[np.arange(len(y)), y]))
        losses.append(loss)

        # 计算准确率
        predictions = np.argmax(y_hat, axis=1)
        accuracy = np.mean(predictions == y)
        accuracies.append(accuracy)

        # Printing out the loss and accuracy at every iteration.
        if epoch % 10 == 0:
            print(f'Epoch {epoch} ==> Loss = {loss}, Accuracy = {accuracy}')

    return w, b, losses, accuracies

# 10个数字的特征矩阵，形状为 (10, 120)
X = np.zeros((10, 120))
for i, letter in enumerate(d.keys()):
    initial_state = np.random.random(1)
    output = []
    for row in d[letter]:
        result = output_row(initial_state, row)
        # 检查 result 的维度
        #print(f"Output row {i}: {result.shape}")  # 打印结果形状，确保它是 (120,) 形状的
        output.append(result)

    # 确保拼接后的长度为120
    output_flat = np.concatenate(output)  # 将每个输出结果拼接成一个大数组
    if len(output_flat) < 120:
        # 如果特征不足，填充
        output_flat = np.pad(output_flat, (0, 120 - len(output_flat)), mode='constant')
    elif len(output_flat) > 120:
        # 如果特征过多，截断
        output_flat = output_flat[:120]

    X[i, :] = output_flat  # 去掉[:, 0]，直接赋值

# 使用 fit 函数进行训练
w, b, losses, accuracies = fit(X, np.arange(10), 0.1, 10, 200)

# 绘制损失曲线和准确率曲线
plt.figure(figsize=(12, 6))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(losses, label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.grid(True)

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(accuracies, label='Accuracy', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epochs')
plt.grid(True)

plt.tight_layout()
plt.show()

def predict(X, w, b):
    """ X --> Input.
    w --> weights.
    b --> bias."""
    # Predicting
    z = X @ w + b
    y_hat = softmax(z)
    # Returning the class with highest probability.
    return np.argmax(y_hat, axis=1)

# 测试并生成混淆矩阵
confusion_matrix = np.zeros((10, 10))

# 用于测试的输入数组（25个测试样本，假设每个样本有120个特征）
X_test = np.zeros((25, 120))

# 假设我们正在测试每个数字（从0到9）
for num_letter in range(10):
    for i in range(5):
        for j in range(5):
            test_letter = d[f'digit_{num_letter}'].copy()
            test_letter[i, j] = 1 if not test_letter[i, j] else 0  # 修改某个位置
            initial_state = np.random.random(1)
            output = []
            case_letter = np.array(test_letter.copy())

            # 处理每一行，获取对应的输出
            for row in case_letter:
                output.append(output_row(initial_state, row))

            # 确保拼接的输出结果符合预期长度（120）
            output_flat = np.concatenate(output)
            if len(output_flat) < 120:
                output_flat = np.pad(output_flat, (0, 120 - len(output_flat)), mode='constant')
            elif len(output_flat) > 120:
                output_flat = output_flat[:120]

            X_test[5 * i + j, :] = output_flat

        # 预测结果
        predictions = predict(X_test, w, b)
        #print(f'Prediction of the corresponding letter {num_letter}: {predictions}')
        for n_lett, prob in Counter(predictions).items():
            confusion_matrix[num_letter, n_lett] = prob / 25 * 100.

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
plt.imshow(confusion_matrix, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(np.arange(10))
plt.yticks(np.arange(10))
plt.show()

