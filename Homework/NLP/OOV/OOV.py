import numpy as np
import jieba
from collections import defaultdict, Counter
import math
import re

# HMM模型类
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

# 加载中文人名语料库并生成训练数据
def load_name_corpus():
    # 假设您已经下载并解压了Chinese-Names-Corpus到指定目录
    # 数据集包含大量的人名列表
    with open('data/Chinese_Names_Corpus（120W）.txt', 'r', encoding='utf-8') as f:
        names = f.read().splitlines()
    # 生成训练数据，标注标签
    train_data = []
    for name in names:
        if len(name.strip()) < 2 or len(name.strip()) > 4:
            continue  # 忽略单字名和超长名字
        name = name.strip()
        sentence = list(name)
        tags = []
        if len(name) == 2:
            tags = ['B', 'E']
        elif len(name) == 3:
            tags = ['B', 'M', 'E']
        else:
            tags = ['B'] + ['M'] * (len(name) - 2) + ['E']
        train_data.append((sentence, tags))
    return train_data

# 主程序
if __name__ == '__main__':
    # 加载训练数据
    train_data = load_name_corpus()
    print("训练数据加载完成，共{}条人名。".format(len(train_data)))

    # 初始化并训练模型
    hmm = HiddenMarkovModel()
    hmm.train(train_data)
    print("模型训练完成。")

    # 测试文本
    test_text = "昨天张三和李四一起来到北京，参加了王五的婚礼。听说欧阳娜娜也在场，和司马光进行了交流。"
    # 分词
    test_sentence = list(test_text.replace(" ", ""))
    # 人名识别
    names = hmm.recognize(test_sentence)
    print("识别出的人名：", names)
