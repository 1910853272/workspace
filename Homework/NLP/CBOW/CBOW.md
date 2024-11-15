# 基于Pytorch框架使用CBOW模型训练中英文预料获得词向量

## 一、CBOW模型

连续词袋模型（CBOW, The Continuous Bag-of-Words Model）是一种典型的**基于上下文的预测模型**，其核心思想是利用周围的上下文词来预测一个中心词。对于每一个单词或词（统称为标识符），使用该标识符周围的标识符来预测当前标识符生成的概率，对于相同的输入，输出每个标识符的概率之和为1。

<img src="https://raw.githubusercontent.com/1910853272/image/master/img/202411142040901.png" alt="1" style="zoom: 33%;" />

### CBOW 的训练过程：

1. **输入：** CBOW 模型将一个上下文窗口的词作为输入。

2. **表示：** 每个词都被表示为一个固定维度的稠密向量（即词嵌入）。

3. **模型架构：**

   - 输入层：上下文词被转换成词向量，并通过求平均来获得一个固定长度的上下文表示。
   - 隐藏层：将上下文词的平均词向量作为输入传递给隐藏层（通常是一个简单的线性变换）。
   - 输出层：通过输出层来预测目标词的概率分布。一般使用 **softmax** 函数将输出转化为概率。

4. **训练目标：** 训练的目标是最大化给定上下文词的情况下预测目标词的概率。这是一个 **条件概率** 问题，即给定上下文词，模型要预测目标词的概率分布。

   训练的损失函数通常是 **交叉熵损失**（cross-entropy loss），用于衡量预测的目标词与实际目标词之间的差距。

<img src="https://raw.githubusercontent.com/1910853272/image/master/img/202411142044765.png" alt="2" style="zoom:33%;" />

## 二、代码实现

### 1.数据预处理

将文本数据分割为单词，并移除停用词（对于中文，还加载停用词文件）。处理后，构建了词汇表，生成了词到索引和索引到词的映射，并准备了上下文-目标词的数据，供CBOW模型训练。

```python
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
```

### 2.定义CBOW模型

该类定义了一个基于CBOW的神经网络模型。模型包括三个部分：

- nn.Embedding：词嵌入层，用于将每个词转换为固定维度的向量。
- nn.Linear：线性投影层，将嵌入向量合并后通过一个128维的投影层。
- 输出层：根据线性投影的结果，预测当前目标词的概率分布。

```python
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
```

### 3.模型训练

```python
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
```

### 4.降维与可视化

使用PCA将训练得到的高维词向量降维到二维，以便可视化。降维后的词向量通过散点图进行展示，每个词通过其对应的词嵌入向量在图中表示。使用plt.annotate在散点图中标注每个词的位置。最后，使用plt.savefig将生成的图像保存为PNG文件。

```python
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
```

### 5.测试模型

在模型训练完成后，提供一个测试接口，可以根据输入的上下文预测目标词。用户输入上下文词，模型返回与之相关性最高的前10个词。

```python
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
```

## 三、实验结果

### 终端输出结果

![2024-11-15_122635](https://raw.githubusercontent.com/1910853272/image/master/img/202411151227336.png)

![terminal](https://raw.githubusercontent.com/1910853272/image/master/img/202411151043967.png)

### 中文可视化结果

![zh_word_vectors](https://raw.githubusercontent.com/1910853272/image/master/img/202411151227708.png)

### 英文可视化结果

![en_word_vectors](https://raw.githubusercontent.com/1910853272/image/master/img/202411151043332.png)