# 汉语未登录词识别算法（Out Of Vocabulary，OOV）

## 1.算法原理

**步骤1：数据准备**

- **训练数据**：包含大量已标注的人名和非人名序列。对于人名部分，按照B、M、E标签进行标注，非人名字符标注为O。

**步骤2：模型训练**

- **计算初始概率 π\piπ**：统计每个状态作为序列起始状态的频率。
- **计算转移概率 AAA**：统计状态间的转移频率，计算概率矩阵。
- **计算发射概率 BBB**：统计每个状态下字符的出现频率，计算发射概率矩阵。

**步骤3：人名识别**

- **输入序列**：待识别的文本序列。
- **分词预处理**：可以选择对文本进行分词，以减少处理长度，提高效率。
- **观测序列**：提取字符序列作为观测序列。
- **维特比解码**：使用训练好的模型参数，对观测序列进行维特比解码，得到最可能的状态序列。
- **结果输出**：根据状态序列，抽取出标记为B-M-E的连续字符序列，识别出人名

## 2.代码

```python
class HiddenMarkovModel:
    def __init__(self):
        # 定义状态集合，B：人名开头，M：人名中间，E：人名结尾，O：其他
        self.states = ['B', 'M', 'E', 'O']
        self.state_num = len(self.states)
        self.observation = set()
        self.start_prob = {}
        self.trans_prob = {}
        self.emit_prob = {}
        self.trained = False

    def train(self, data):
        # 初始化概率矩阵
        state_count = defaultdict(int)
        trans_count = defaultdict(lambda: defaultdict(int))
        emit_count = defaultdict(lambda: defaultdict(int))

        for sentence, tags in data:
            for i in range(len(tags)):
                state = tags[i]
                observation = sentence[i]
                state_count[state] += 1
                emit_count[state][observation] += 1
                self.observation.add(observation)
                if i == 0:
                    self.start_prob[state] = self.start_prob.get(state, 0) + 1
                else:
                    prev_state = tags[i - 1]
                    trans_count[prev_state][state] += 1

        # 计算初始概率
        total_start = sum(self.start_prob.values())
        for state in self.start_prob:
            self.start_prob[state] /= total_start

        # 计算转移概率
        self.trans_prob = {state: {} for state in self.states}
        for prev_state in trans_count:
            total = sum(trans_count[prev_state].values())
            for state in trans_count[prev_state]:
                self.trans_prob[prev_state][state] = trans_count[prev_state][state] / total

        # 计算发射概率
        self.emit_prob = {state: {} for state in self.states}
        for state in emit_count:
            total = sum(emit_count[state].values())
            for obs in emit_count[state]:
                self.emit_prob[state][obs] = emit_count[state][obs] / total

        self.trained = True

    def viterbi(self, sentence):
        V = [{}]
        path = {}

        for state in self.states:
            V[0][state] = self.start_prob.get(state, 1e-6) * self.emit_prob[state].get(sentence[0], 1e-6)
            path[state] = [state]

        for t in range(1, len(sentence)):
            V.append({})
            new_path = {}

            for y in self.states:
                (prob, state) = max(
                    [(V[t - 1][y0] * self.trans_prob.get(y0, {}).get(y, 1e-6) * self.emit_prob[y].get(sentence[t], 1e-6), y0) for y0 in self.states])
                V[t][y] = prob
                new_path[y] = path[state] + [y]

            path = new_path

        n = len(sentence) - 1
        (prob, state) = max([(V[n][y], y) for y in self.states])
        return path[state]

    def recognize(self, sentence):
        if not self.trained:
            raise Exception("模型未训练，请先调用train方法训练模型。")
        tags = self.viterbi(sentence)
        names = []
        name = ''
        for char, tag in zip(sentence, tags):
            if tag == 'B':
                if name:
                    names.append(name)
                    name = ''
                name += char
            elif tag == 'M':
                name += char
            elif tag == 'E':
                name += char
                names.append(name)
                name = ''
            else:
                if name:
                    names.append(name)
                    name = ''
        if name:
            names.append(name)
        # 过滤非人名的部分
        filtered_names = [n for n in names if len(n) >= 2]
        return filtered_names
```

## 3.结果

![2024-11-11_224607](https://raw.githubusercontent.com/1910853272/image/master/img/202411112246675.png)

上述训练结果并不好

**优点：**

- **模型简单**：HMM模型结构清晰，易于理解和实现。
- **适用于未登录词**：通过统计字符和标签的概率关系，能够识别未在训练集中出现过的人名。
- **计算效率高**：维特比算法具有线性时间复杂度，适合处理长文本。

**缺点：**

- **假设独立性**：HMM假设当前状态仅与前一状态相关，可能无法捕获长距离依赖。
- **特征单一**：只利用了字符序列信息，未考虑其他特征，如词性、上下文等。
- **数据稀疏性**：在发射概率和转移概率的计算中，可能会遇到数据稀疏问题，需要平滑处理。
