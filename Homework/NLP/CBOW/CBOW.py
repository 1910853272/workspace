import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm, trange

# 初始化随机种子，使得每次运行的结果相同
torch.manual_seed(1)  # 这个可以保留，确保结果可复现

# 加载停用词，用于处理文本时去除无意义的词（如 "的", "是", "在" 等）
def load_stop_words():
    """
        停用词是指在信息检索中，
        为节省存储空间和提高搜索效率，
        在处理自然语言数据（或文本）之前或之后
        会自动过滤掉某些字或词
    """
    with open('./data/stopwords.txt', "r", encoding="utf-8") as f:
        return f.read().split("\n")  # 返回停用词列表


# 输入模块，加载和处理文本数据
def input(mode):
    print("\n=================1.数据预处理阶段=======================")
    raw_text = []  # 用于存储处理后的文本

    # 中文数据处理
    if mode == "zh":
        stop_words = load_stop_words()  # 加载停用词
        with open('./data/zh.txt', encoding="utf-8") as f:
            data = f.read()
            text = data.split()  # 以空格分割文本
            # 遍历文本，去除停用词
            for word in text:
                if word not in stop_words:
                    raw_text.append(word)
    # 英文数据处理
    else:
        with open('./data/en.txt') as f:
            data = f.read()
        raw_text = data.split()  # 以空格分割文本

    print("raw_text=", raw_text)  # 打印处理后的文本

    # 构建词汇表，去重后的词汇表
    vocab = set(raw_text)
    vocab_size = len(vocab)  # 词汇表大小
    # 单词到索引的映射
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    # 索引到单词的映射
    idx_to_word = {i: word for i, word in enumerate(vocab)}

    # 使用滑动窗口方法生成训练数据
    data = []  # 存储上下文词和目标词
    for i in range(2, len(raw_text) - 2):  # 滑动窗口，取上下文为左右2个词
        context = [raw_text[i - 2], raw_text[i - 1], raw_text[i + 1], raw_text[i + 2]]
        target = raw_text[i]
        data.append((context, target))  # 数据格式为([上下文], 目标词)

    print(data[:5])  # 打印前5条训练数据
    return vocab_size, word_to_idx, idx_to_word, data  # 返回处理后的数据和词汇表


# 将上下文词转换为索引向量
def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]  # 将每个单词转换为对应的索引
    return torch.tensor(idxs, dtype=torch.long)  # 转换为Tensor对象，方便后续处理


# 定义CBOW模型
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)  # 词嵌入层
        self.proj = nn.Linear(embedding_dim, 128)  # 线性投影层
        self.output = nn.Linear(128, vocab_size)  # 输出层，用于预测目标词

    def forward(self, inputs):
        # 输入词的词向量，通过embedding层查找
        embeds = sum(self.embeddings(inputs)).view(1, -1)  # 将上下文词的词向量求和，并调整维度
        out = F.relu(self.proj(embeds))  # 线性投影后应用ReLU激活
        out = self.output(out)  # 通过输出层预测目标词
        nll_prob = F.log_softmax(out, dim=-1)  # 通过log_softmax计算对数概率
        return nll_prob  # 返回对数概率


# 训练函数
def train():
    print("=================2.模型训练阶段=======================")
    for epoch in trange(epochs):  # 遍历所有训练轮次
        total_loss = 0  # 初始化总损失
        for context, target in tqdm(data):  # 遍历训练数据集
            # 获取上下文词的索引向量，并将其移至指定的设备（GPU或CPU）
            context_vector = make_context_vector(context, word_to_idx).to(device)
            target = torch.tensor([word_to_idx[target]])  # 将目标词转换为索引并转为tensor

            model.zero_grad()  # 清除之前计算的梯度
            train_predict = model(context_vector)  # 前向传播得到预测结果
            loss = loss_function(train_predict, target)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            total_loss += loss.item()  # 累加损失

        losses.append(total_loss)  # 记录每轮的总损失

    print("losses-=", losses)  # 打印所有轮次的损失
    W = model.embeddings.weight.cpu().detach().numpy()  # 获取训练后的词向量矩阵
    return W  # 返回词向量矩阵


# 输出处理函数，保存词向量到文件
def output(W, mode):
    print("=================4.输出处理=======================")
    word_2_vec = {}  # 用于存储词到词向量的映射
    for word in word_to_idx.keys():
        word_2_vec[word] = W[word_to_idx[word], :]  # 根据词汇表中的索引从词向量矩阵中获取词向量

    # 根据模式选择文件编码格式
    if mode == "zh":
        with open("./output/zh_wordvec.txt", 'w', encoding='utf-8') as f:
            for key in word_to_idx.keys():
                f.write('\n')
                f.writelines('"' + str(key) + '":' + str(word_2_vec[key]))
            f.write('\n')
    else:
        with open("./output/en_wordvec.txt", 'w') as f:
            for key in word_to_idx.keys():
                f.write('\n')
                f.writelines('"' + str(key) + '":' + str(word_2_vec[key]))
            f.write('\n')

    print("词向量已保存")  # 提示词向量已保存


# 可视化词向量，使用PCA降维到2D并绘制图形
def show(W, mode):
    print("=================5.可视化阶段=======================")
    pca = PCA(n_components=2)  # PCA降维到二维
    principalComponents = pca.fit_transform(W)  # 对词向量矩阵进行降维
    word2ReduceDimensionVec = {}  # 存储降维后的词向量

    # 将词向量降维后存储
    for word in word_to_idx.keys():
        word2ReduceDimensionVec[word] = principalComponents[word_to_idx[word], :]

    plt.figure(figsize=(20, 20))  # 创建大图，避免重叠
    count = 0  # 计数器，限制显示1000个词
    if mode == "zh":  # 如果是中文模式
        for word, wordvec in word2ReduceDimensionVec.items():
            if count < 1000:  # 只显示前1000个词
                plt.rcParams['font.sans-serif'] = ['SimHei']  # 处理中文显示问题
                plt.rcParams['axes.unicode_minus'] = False  # 处理负号显示问题
                plt.scatter(wordvec[0], wordvec[1])  # 绘制词向量
                plt.annotate(word, (wordvec[0], wordvec[1]))  # 注释词语
                count += 1
    else:  # 如果是英文模式
        for word, wordvec in word2ReduceDimensionVec.items():
            if count < 1000:
                plt.scatter(wordvec[0], wordvec[1])
                plt.annotate(word, (wordvec[0], wordvec[1]))
                count += 1
    plt.show()  # 显示绘图


# 测试函数，输入上下文，预测目标词
def test(mode):
    print("=================3.测试阶段=======================")

    # 根据模式选择上下文词
    if mode == "zh":
        context = ['粮食', '出现', '过剩', '恰好']
    else:
        context = ['present', 'food', 'can', 'specifically']

    # 将上下文转换为索引向量
    context_vector = make_context_vector(context, word_to_idx).to(device)
    predict = model(context_vector).data.cpu().numpy()  # 获取预测结果
    print('Test Context: {}'.format(context))
    max_idx = np.argmax(predict)  # 找到预测的词
    print('Prediction: {}'.format(idx_to_word[max_idx]))  # 输出预测词
    print("CBOW embedding'weight=", model.embeddings.weight)  # 打印词向量


# 主函数
if __name__ == '__main__':
    # -------------------0.参数设置--------------------#
    mode = "en"  # 设置为英文模式
    learning_rate = 0.001  # 学习率
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设置设备（GPU/CPU）
    context_size = 2  # 上下文窗口大小
    embedding_dim = 100  # 词向量维度
    epochs = 10  # 训练轮数
    losses = []  # 用于记录每轮的损失
    loss_function = nn.NLLLoss()  # 负对数似然损失函数

    # -------------------1.输入与预处理模块---------------------#
    vocab_size, word_to_idx, idx_to_word, data = input(mode)

    # -------------------2.训练模块---------------------------#
    model = CBOW(vocab_size, embedding_dim).to(device)  # 实例化模型并移至设备
    optimizer = optim.SGD(model.parameters(), lr=0.001)  # 设置优化器为SGD

    W = train()  # 训练模型

    # -------------------3.测试模块---------------------------#
    test(mode)

    # -------------------4.输出处理模块------------------------#
    output(W, mode)

    # -------------------5.可视化模块-------------------------#
    show(W, mode)
