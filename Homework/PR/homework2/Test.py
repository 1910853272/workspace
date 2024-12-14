import pandas as pd
import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt

# 从txt中读取数据集，并把标签转化成one-hot矩阵
class readData(object):

    def __init__(self, io='Data.xlsx'):
        """
        io:数据集路径 excel格式
        """
        df = pd.read_excel(io)
        all_data = df.values  # 所有数据 特征+标签
        permutation = np.random.permutation(all_data.shape[0])
        all_data = all_data[permutation, :]
        self.data = all_data[:, 0:3]  # 提取特征集
        self.label = all_data[:, 3]  # 提取标签

    def get_train_data(self):
        train_data = np.hstack((np.ones((self.data.shape[0], 1)), self.data))  # 每个样本的特征最前面都插入1维阈值 1(偏置)
        train_label = self.label.reshape(-1).astype(int) - 1  # 类别标签从1-3变为0-2，便于转成onw-hot矩阵进行计算
        # print(train_label)
        train_y = np.eye(3)[train_label]  # one-hot向量矩阵

        return train_data, train_y

# 激励函数及其导数
class stimulateFunc(object):

    def __init__(self):
        """初始化函数 没有任何操作"""
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.power(self.tanh(x), 2)

# 输入层：隐含层结点的激励函数采用双曲正切函数
class inputLayer(object):

    def __init__(self, input_num=4, hidden_num=3, learning_rate=0.1):
        """
        搭建神经网络输入层到隐含层的映射
        采用tanh函数激励
        需要参数：
        X:输入的特征矩阵
        input_num:输入数据维度
        hidden_num:隐含层节点数
        learning_rate:学习率/更新步长
        W_ih:权重矩阵
        Y_h:激活后输出到隐含层的特征矩阵
        """
        self.X = []  # 输入矩阵
        self.input_num = input_num  # 需要注意，输入要多一个偏置项
        self.hidden_num = hidden_num
        self.learning_rate = learning_rate

        self.W_ih = np.random.rand(self.input_num, self.hidden_num)  # 随机初始化权重 [0,1)之间 包括偏置的权重
        self.net = []  # 输入矩阵*当前权重所得节点
        self.Y_h = []

    def forward(self, X):  # X:输入矩阵(特征+1维偏置)
        """前向传播"""
        self.X = X
        self.net = np.dot(self.X, self.W_ih)  # 乘上权重矩阵
        self.Y_h = stimulateFunc().tanh(self.net)  # 激励结果
        return self.Y_h  # 返回激励结果

    def backward(self, delta):
        """反向传播"""
        """delta:后一层收集的误差"""
        temp = stimulateFunc().tanh_derivative(self.net)
        d_W_ih = np.dot(self.X.T, (delta * temp))  # 计算更新量
        self.W_ih = self.W_ih + self.learning_rate * d_W_ih  # 更新权重
        return self.W_ih  # 返回更新后的权重

# 隐藏层：输出层的激励函数采用 sigmoid 函数
class hiddenLayer(object):

    def __init__(self, hidden_num=3, output_num=3, learning_rate=0.1):
        """
        搭建神经网络隐含层到输出层的映射
        采用sigmoid函数激励
        需要参数：
        Y:输入层传进来的特征矩阵
        hidden_num:隐含层节点数
        output_num:输出层节点数
        learning_rate:学习率/更新步长
        W_ho:权重矩阵
        Z_o:输出到输出层的特征矩阵
        """
        self.hidden_num = hidden_num  # 获得隐含层节点数
        self.output_num = output_num
        self.learning_rate = learning_rate

        self.W_ho = np.random.rand(self.hidden_num, self.output_num)  # 随机初始化权重 [0,1)之间 包括偏置的权重
        self.net = []
        self.Z_o = []

    def forward(self, Y):  # Y:上一层输入矩阵
        """前向传播"""
        self.Y = Y
        self.net = np.dot(Y, self.W_ho)  # 乘上权重矩阵
        # print(temp)
        self.Z_o = stimulateFunc().sigmoid(self.net)  # 激励结果
        return self.Z_o  # 返回激励结果

    def backward(self, delta):
        """反向传播"""
        """delta:后一层收集的误差"""
        temp = stimulateFunc().sigmoid_derivative(self.net)
        delta_ho = delta * temp
        d_W_ho = np.dot(self.Y.T, delta_ho)  # 计算更新量
        self.W_ho = self.W_ho + self.learning_rate * d_W_ho  # 更新权重
        back_delta = np.dot(delta_ho, self.W_ho.T)
        return back_delta  # 返回传到前一层的误差

# 三层前向神经网络反向传播算法：采用批量方式更新权重
# 目标函数采用平方误差准则函数MSE
class bp_model(object):
    def __init__(self, X, y, epoch, learning_rate, batch_size, hidden_num):
        """
        X:特征
        y:标签（one-hot矩阵
        epoch：迭代次数
        learning_rate：更新步长/学习率
        batch_size:批量更新数。为1则为单样本更新
        hidden_num：隐含层节点数
        """
        self.X = X
        self.y = y
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.hidden_num = hidden_num

        self.input_num = X.shape[1]
        self.output_num = y.shape[1]

    def train(self):
        X_ = self.X
        Y_ = self.y
        loss_ = []
        acc_ = []
        epoch_ = 0
        bp = BP_once(self.input_num, self.output_num, self.hidden_num, self.learning_rate)  # 搭建神经网络
        for i in range(self.epoch):
            for j in range(X_.shape[0] // self.batch_size):  # 批量更新大小
                x_ = X_[j * self.batch_size:(j + 1) * self.batch_size, :]
                y_ = Y_[j * self.batch_size:(j + 1) * self.batch_size, :]
                bp.training(x_, y_)
            acc = bp.acc(X_, Y_)
            loss = bp.MSEloss(X_, Y_)
            acc_.append(acc)
            loss_.append(loss)
            # if i % 100 == 0:
            #     print('epoch=', i)
            #     print('loss = %.10f, acc=%.1f' % (loss, acc))
            if acc == 100:
                epoch_ = i  # 记录下训练完全正确的迭代次数
            if loss <= 0.01:
                print('>>>>>loss has already down to 0.01! Training Terminated!<<<<<')
                break
        return loss_, acc_, epoch_

# 三层前向神经网络反向传播算法：单样本方式更新权重
# 目标函数采用平方误差准则函数MSE
class BP_once(object):
    def __init__(self, input_num=4, output_num=3, hidden_num=3, learning_rate=0.01):
        """
        构造三层前向神经网络
        进行一次反向传播更新权重
        误差准则 MSE
        所需参数：
        iteration_num：迭代次数
        learning_rate：更新步长/学习率
        hidden_num：隐含层节点数
        """
        self.input_num = input_num
        self.hidden_num = hidden_num
        self.output_num = output_num
        self.learning_rate = learning_rate
        # 搭建神经网络
        self.input = inputLayer(self.input_num, self.hidden_num, self.learning_rate)
        self.hidden = hiddenLayer(self.hidden_num, self.output_num, self.learning_rate)

    # 误差函数 MSE准则
    def MSEloss(self, X, Y):
        return np.sum(np.power(self.predict(X) - Y, 2) / 2)

    # 预测准确率
    def acc(self, X, Y):
        count = (np.sum(np.argmax(Y, axis=1) == np.argmax(self.predict(X), axis=1)))
        return count / X.shape[0] * 100

    # 前向传播获得预测情况
    def predict(self, X):
        x = X
        y = self.input.forward(x)
        z = self.hidden.forward(y)
        return z

    # 反向传播更新权重
    def update(self, X, y):
        z = self.predict(X)  # 一次前向传播
        delta = y - z  # 计算最后一层误差
        delta = self.hidden.backward(delta)  # 向前传播误差
        self.input.backward(delta)
        return 1

    def training(self, X, y):
        """进行一次前向-反向传播"""
        # self.model_builder()
        self.update(X, y)
        # loss = self.MSEloss()
        # acc = self.acc()
        return 1

# 主函数，绘制acc和loss图像
if __name__ == '__main__':
    np.random.seed(10)
    data = readData()
    X, y = data.get_train_data()

    hidden_num = [3, 5, 10, 20]
    batch_size = [1, 5, 10]
    learning_rate = [0.01, 0.05, 0.1, 1]
    running_time = []
    stop_epoch = []
    each_loss = []
    each_acc = []

    # """开始计算运行时间"""
    # for i in range(len(learning_rate)):
    #
    #     start = time.time()
    #     my_model = bp_model(X, y, epoch=5000, batch_size=batch_size[0],
    #                         learning_rate=learning_rate[i], hidden_num=hidden_num[2])
    #     loss, acc, full_epoch = my_model.train()
    #     each_acc.append(acc[-1])
    #     each_loss.append(loss[-1])
    #     end = time.time()
    #     running_time.append(end - start)
    #     stop_epoch.append(full_epoch)
    #     # print('time cost : %.5f sec' % running_time[i])
    #     # print('epoch reach to %d, acc=100' % stop_epoch[i])
    #
    # print('time cost : ', running_time)
    # print('stop_epoch: ', stop_epoch)
    # print('loss: ', each_loss)
    # print('acc: ', each_acc)
    # # print(stop_epoch)

    """分别绘制acc和loss图像"""
    my_model = bp_model(X, y, epoch=5000, batch_size=batch_size[1],
                        learning_rate=learning_rate[2], hidden_num=hidden_num[2])
    loss, acc, full_epoch = my_model.train()
    # 美化后的图表
    plt.figure(1)
    plt.plot(loss, color='b', linestyle='-', linewidth=2, label='Loss')  # 使用蓝色线条表示loss
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # 更细的网格
    plt.title('Training Loss Over Epochs', fontsize=16)  # 标题
    plt.xlabel('Epoch', fontsize=14)  # x轴标签
    plt.ylabel('Loss', fontsize=14)  # y轴标签
    plt.axis('tight')
    plt.legend()  # 添加图例
    plt.xticks(fontsize=12)  # 设置x轴字体大小
    plt.yticks(fontsize=12)  # 设置y轴字体大小
    plt.tight_layout()  # 自动调整布局

    plt.figure(2)
    plt.plot(acc, color='g', linestyle='-', linewidth=2, label='Accuracy')  # 使用绿色线条表示accuracy
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # 更细的网格
    plt.title('Training Accuracy Over Epochs', fontsize=16)  # 标题
    plt.xlabel('Epoch', fontsize=14)  # x轴标签
    plt.ylabel('Accuracy (%)', fontsize=14)  # y轴标签
    plt.axis('tight')
    plt.legend()  # 添加图例
    plt.xticks(fontsize=12)  # 设置x轴字体大小
    plt.yticks(fontsize=12)  # 设置y轴字体大小
    plt.tight_layout()  # 自动调整布局

    plt.show()

