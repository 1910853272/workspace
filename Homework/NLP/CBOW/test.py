import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm, trange

# 初始化矩阵
torch.manual_seed(1)  # 这个还不知道要不要删除


def load_stop_words():
    """
        停用词是指在信息检索中，
        为节省存储空间和提高搜索效率，
        在处理自然语言数据（或文本）之前或之后
        会自动过滤掉某些字或词
    """
    with open('../data/stopwords.txt', "r", encoding="utf-8") as f:
        return f.read().split("\n")


def input(mode):
    print("\n=================1.数据预处理阶段=======================")
    raw_text = []

    if mode == "zh":
        stop_words = load_stop_words()
        with open('./data/zh_test.txt', encoding="utf-8") as f:
            data = f.read()
            text = data.split()                              # 以空格分割
            for word in text:                                # 数组遍历
                if word not in stop_words:                   # 中文语料需要去掉一些停止词
                    raw_text.append(word)
    else:
        with open('./data/en.txt') as f:
            data = f.read()
        raw_text = data.split()                              # 以空格分割

    print("raw_text=", raw_text)                             # raw_text数组存储分割结果

    vocab = set(raw_text)                                    # 删除重复元素/单词
    vocab_size = len(vocab)                                  # 这里的size是去重之后的词表大小
    word_to_idx = {word: i for i, word in enumerate(vocab)}  # 由单词索引下标
    idx_to_word = {i: word for i, word in enumerate(vocab)}  # 由下标索引单词

    data = []                                                # cbow那个词表，即{[w1,w2,w4,w5],"label"}这样形式
    for i in range(2, len(raw_text) - 2):  # 类似滑动窗口
        context = [raw_text[i - 2], raw_text[i - 1],
                   raw_text[i + 1], raw_text[i + 2]]
        target = raw_text[i]
        data.append((context, target))

    print(data[:5])  # (['the', 'present', 'surplus', 'can'], 'food')
    return vocab_size, word_to_idx, idx_to_word, data


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)


# 模型结构
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.proj = nn.Linear(embedding_dim, 128)
        self.output = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = sum(self.embeddings(inputs)).view(1, -1) # 这里为什么要求和啊？a=embedding(input)是去embedding.weight中取对应index的词向量！
        out = F.relu(self.proj(embeds))
        out = self.output(out)
        nll_prob = F.log_softmax(out, dim=-1)
        return nll_prob


def train():
    print("=================2.模型训练阶段=======================")
    for epoch in trange(epochs):
        total_loss = 0
        for context, target in tqdm(data):
            context_vector = make_context_vector(context, word_to_idx).to(device)  # 把训练集的上下文和标签都放到cpu中
            target = torch.tensor([word_to_idx[target]])
            model.zero_grad()                                                      # 梯度清零
            train_predict = model(context_vector)                                  # 开始前向传播
            loss = loss_function(train_predict, target)
            loss.backward()                                                        # 反向传播
            optimizer.step()                                                       # 更新参数
            total_loss += loss.item()
        losses.append(total_loss)                                                  # 更新损失

    print("losses-=", losses)
    W = model.embeddings.weight.cpu().detach().numpy()
    return W


def output(W, mode):
    print("=================4.输出处理=======================")
    word_2_vec = {}  # 生成词嵌入字典，即{单词1:词向量1,单词2:词向量2...}的格式
    for word in word_to_idx.keys():
        word_2_vec[word] = W[word_to_idx[word], :]  # 词向量矩阵中某个词的索引所对应的那一列即为所该词的词向量

    # 将生成的字典写入到文件中
    if mode == "zh":
        with open("../output/en_wordvec.txt", 'w', encoding='utf-8') as f:  # 中文字符集要设置为utf-8，不然会乱码
            for key in word_to_idx.keys():
                f.write('\n')
                f.writelines('"' + str(key) + '":' + str(word_2_vec[key]))
            f.write('\n')
    else:
        with open("../output/en_wordvec.txt", 'w') as f:
            for key in word_to_idx.keys():
                f.write('\n')
                f.writelines('"' + str(key) + '":' + str(word_2_vec[key]))
            f.write('\n')

    print("词向量已保存")


def show(W,mode):  # 将词向量降成二维之后，在二维平面绘图
    print("=================5.可视化阶段=======================")
    pca = PCA(n_components=2)  # 数据降维
    principalComponents = pca.fit_transform(W)
    word2ReduceDimensionVec = {}  # 降维后在生成一个词嵌入字典，即即{单词1:(维度一，维度二),单词2:(维度一，维度二)...}的格式

    for word in word_to_idx.keys():
        word2ReduceDimensionVec[word] = principalComponents[word_to_idx[word], :]

    plt.figure(figsize=(20, 20))  # 将词向量可视化
    count = 0
    if mode=="zh":
        for word, wordvec in word2ReduceDimensionVec.items():
            if count < 1000:
                plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
                plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号，否则负号会显示成方块
                plt.scatter(wordvec[0], wordvec[1])
                plt.annotate(word, (wordvec[0], wordvec[1]))
                count += 1
    else:
        for word, wordvec in word2ReduceDimensionVec.items():
            if count < 1000:  # 只画出1000个，太多显示效果很差
                plt.scatter(wordvec[0], wordvec[1])
                plt.annotate(word, (wordvec[0], wordvec[1]))
                count += 1
    plt.show()



def test(mode):
    print("=================3.测试阶段=======================")

    if mode =="zh":
        context = ['粮食', '出现', '过剩', '恰好']
    else:
        context = ['present', 'food', 'can', 'specifically']

    context_vector = make_context_vector(context, word_to_idx).to(device)
    predict = model(context_vector).data.cpu().numpy()  # 预测的值
    # print('Raw text: {}\n'.format(' '.join(raw_text)))
    print('Test Context: {}'.format(context))
    max_idx = np.argmax(predict)  # 返回最大值索引
    print('Prediction: {}'.format(idx_to_word[max_idx]))  # 输出预测的值
    print("CBOW embedding'weight=", model.embeddings.weight)  # 获取词向量，这个Embedding就是我们需要的词向量，他只是一个模型的一个中间过程


if __name__ == '__main__':
    # -------------------0.参数设置--------------------#
    # 训练中文时注释掉英文即可
    #mode = "zh"
    mode = "en"

    learning_rate = 0.001                                                       # 学习率 超参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       # 放cuda或者cpu里，我的电脑不支持cuda
    context_size = 2                                                            # 上下文信息，上下文各2个词，中心词作为标签
    embedding_dim = 100                                                         # 词向量维度，一般都是要100-300个之间
    epochs = 10                                                                 # 训练次数
    losses = []                                                                 # 存储损失的集合
    loss_function = nn.NLLLoss()

    # -------------------1.输入与预处理模块---------------------#
    vocab_size, word_to_idx, idx_to_word, data = input(mode)

    # -------------------2.训练模块---------------------------#
    model = CBOW(vocab_size, embedding_dim).to(device)          # 模型在cup训练
    optimizer = optim.SGD(model.parameters(), lr=0.001)         # 优化器

    W = train()
    # -------------------3.测试模块---------------------------#
    test(mode)
    # -------------------4.输出处理模块------------------------#
    output(W, mode)
    # -------------------5.可视化模块-------------------------#
    show(W, mode)