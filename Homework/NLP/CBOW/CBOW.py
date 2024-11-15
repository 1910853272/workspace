import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm, trange
import matplotlib

# 设置中文字体，避免中文乱码
matplotlib.rcParams['font.family'] = 'SimHei'  # 或者选择你系统中已有的其他字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置随机种子，以保证每次运行时结果相同
torch.manual_seed(1)

# 加载停用词
def load_stop_words(file_path='./data/stopwords.txt'):
    """读取停用词文件，返回停用词列表"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().splitlines()

# 文本预处理函数
def preprocess_text(file_path, mode="en"):
    """
    处理输入文本，将其分割为单词，并去除停用词
    :param file_path: 输入文本文件路径
    :param mode: 'zh' 为中文，'en' 为英文
    :return: 词汇表大小，词到索引的映射，索引到词的映射，处理后的数据
    """
    raw_text = []  # 用于存储处理后的文本

    if mode == "zh":
        stop_words = load_stop_words()  # 加载中文停用词
        with open(file_path, encoding="utf-8") as f:
            text = f.read().split()  # 以空格分割文本
            raw_text = [word for word in text if word not in stop_words]  # 去除停用词
    else:
        with open(file_path) as f:
            raw_text = f.read().split()  # 以空格分割英文文本

    print("raw_text=", raw_text[:10])  # 打印处理后的前10个单词

    # 构建词汇表
    vocab = set(raw_text)
    vocab_size = len(vocab)
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    # 生成上下文-目标数据
    data = []
    for i in range(2, len(raw_text) - 2):
        context = [raw_text[i - 2], raw_text[i - 1], raw_text[i + 1], raw_text[i + 2]]
        target = raw_text[i]
        data.append((context, target))

    print(f"Processed data sample: {data[:5]}")
    return vocab_size, word_to_idx, idx_to_word, data

# 将上下文词转换为索引
def make_context_vector(context, word_to_idx):
    """将上下文单词转换为索引列表"""
    return torch.tensor([word_to_idx[word] for word in context], dtype=torch.long)

# 定义CBOW模型
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)  # 词嵌入层
        self.proj = nn.Linear(embedding_dim, 128)  # 线性投影层
        self.output = nn.Linear(128, vocab_size)  # 输出层

    def forward(self, inputs):
        # 将上下文词的嵌入向量求和并投影到128维空间，再通过输出层预测目标词
        embeds = self.embeddings(inputs).sum(dim=0).view(1, -1)  # 求和后调整维度
        out = F.relu(self.proj(embeds))  # 线性投影后通过ReLU激活
        out = self.output(out)  # 输出层计算目标词的对数概率
        return F.log_softmax(out, dim=-1)

# 训练函数
def train(model, data, loss_function, optimizer, device):
    """训练模型，返回训练后的词向量"""
    losses = []
    model.train()
    for epoch in trange(epochs):
        total_loss = 0
        for context, target in tqdm(data):
            context_vector = make_context_vector(context, word_to_idx).to(device)
            target = torch.tensor([word_to_idx[target]], dtype=torch.long).to(device)

            optimizer.zero_grad()
            prediction = model(context_vector)
            loss = loss_function(prediction, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        losses.append(total_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss}")

    # 获取训练后的词向量矩阵
    W = model.embeddings.weight.cpu().detach().numpy()
    return W, losses

# 保存词向量到文件
def save_word_vectors(W, word_to_idx, mode="en"):
    """保存词向量到文件"""
    word_2_vec = {word: W[idx] for word, idx in word_to_idx.items()}

    file_path = f"./output/{mode}_wordvec.txt"
    with open(file_path, 'w', encoding='utf-8') as f:
        for word, vec in word_2_vec.items():
            f.write(f'"{word}": {vec.tolist()}\n')
    print(f"Word vectors saved to {file_path}")

# 使用PCA降维并可视化词向量
def visualize_word_vectors(W, word_to_idx, mode="en"):
    """使用PCA将词向量降维到2D并可视化"""
    pca = PCA(n_components=2)
    reduced_vecs = pca.fit_transform(W)

    plt.figure(figsize=(16, 16))

    for idx, word in enumerate(word_to_idx.keys()):
        x, y = reduced_vecs[idx]
        plt.scatter(x, y, marker='o', color='b', alpha=0.5)
        plt.annotate(word, (x, y), fontsize=10, alpha=0.7)

    plt.title(f"Word Vectors ({mode})", fontsize=20)
    plt.xlabel('PCA Component 1', fontsize=14)
    plt.ylabel('PCA Component 2', fontsize=14)

    # Save the image to a file
    plt.savefig(f'./output/{mode}_word_vectors.png', bbox_inches='tight', dpi=300)
    plt.show()
    print(f"Word vector visualization saved to ./output/{mode}_word_vectors.png")

# 测试函数，根据输入的上下文预测目标词
def test(model, word_to_idx, device):
    """根据上下文词，预测目标词"""
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        while True:
            context_input = input("请输入上下文：").split()
            context_vector = make_context_vector(context_input, word_to_idx).to(device)
            prediction = model(context_vector)
            top_k = torch.topk(prediction, 10)
            print(f"Top 10 predictions: {top_k}")

# 主函数
if __name__ == '__main__':
    # 参数设置
    mode = "zh"  # 可选择 'zh' 或 'en'
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选择设备
    embedding_dim = 100
    epochs = 10

    # 加载并处理数据
    vocab_size, word_to_idx, idx_to_word, data = preprocess_text('./data/zh.txt', mode=mode) # 可选择 'zh' 或 'en'

    # 创建模型和优化器
    model = CBOW(vocab_size, embedding_dim).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = nn.NLLLoss()

    # 训练模型
    W, losses = train(model, data, loss_function, optimizer, device)

    # 保存词向量
    save_word_vectors(W, word_to_idx, mode)

    # 可视化词向量并保存
    visualize_word_vectors(W, word_to_idx, mode)

    # 测试模型
    test(model, word_to_idx, device)
