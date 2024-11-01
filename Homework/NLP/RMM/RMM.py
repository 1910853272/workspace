# -*- coding:utf-8 -*-
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn

# 设置绘图风格
plt.style.use('fivethirtyeight')

def readFile(filename):
    '''
    读取文件并返回一个生成器，每个元素是一行文本

    Parameters
    ----------
        filename: str, 文件名

    Returns
    ----------
        generator: 每次返回一行文本
    '''
    with open(filename, 'r', encoding='utf-8') as f:
        # 读取文件的第一行
        data = f.readline().strip()
        while data:
            yield data  # 返回当前行数据
            try:
                # 继续读取下一行
                data = f.readline().strip()
            except:
                # 如果读取失败，打印错误信息并继续
                print('read file failed in one line!')
                continue

def preprocess_text(text):
    '''
    去掉每个词语后面的标注，只保留词语本身

    Parameters
    ----------
        text: str, 输入文本，例如：'９９/m 昆明/ns 世博会/n 组委会/j 秘书长/n 、/w 云南省/ns'

    Returns
    ----------
        str: 预处理后的文本，例如：'９９昆明世博会组委会秘书长、云南省'
    '''
    # 使用lambda函数处理每个词语，去掉词性标注部分
    return ''.join(map(lambda x: x[:x.rfind('/')], text.split('  ')))

def preprocess_text2(text):
    '''
    保留每个词语的标注信息

    Parameters
    ----------
        text: str, 输入文本，例如：'９９/m 昆明/ns 世博会/n 组委会/j 秘书长/n 、/w 云南省/ns'

    Returns
    ----------
        str: 保留词性标注的文本，例如：'/９９/昆明/世博会/组委会/秘书长/、/云南省'
    '''
    # 使用lambda函数处理每个词语，保留词性标注部分
    return ''.join(map(lambda x: x[:x.rfind('/')+1], text.split('  ')))

def precision_recall_f1(output, target):
    '''
    计算分词结果的精确度、召回率和 F1 分数

    Parameters
    ----------
        output: str, 输出的分词结果
        target: str, 目标分词结果

    Returns
    ----------
        precision: float, 精确度
        recall: float, 召回率
        f1: float, F1 分数
    '''
    def extract_index_pair(text):
        # 提取每个词语的起始和结束索引对
        o = [(0, 0)]
        index = 0
        for i in text:
            if i != '/':
                index += 1  # 如果不是'/'，索引递增
            else:
                o.append((o[-1][-1], index))  # 遇到'/'时记录索引对
        else:
            o.append((o[-1][-1], index))
        o = set(o)  # 转换为集合
        o.remove((0, 0))  # 移除初始值
        return o

    o = extract_index_pair(output)
    t = extract_index_pair(target)

    def precision_score(o, t):
        # 计算精确度
        count = 0
        for i in t:
            if i in o:
                count += 1
        return count / len(t)

    # 计算精确度、召回率和 F1 分数
    precision, recall = precision_score(o, t), precision_score(t, o)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1

def build_corpus_and_testing_text(filename, training_ratio=0.7):
    '''
    构建词库和测试文本

    Parameters
    ----------
    filename: str, 文件名
    training_ratio: float, 训练数据的比例

    Returns
    ----------
    corpus: set, 构建的词库
    testing_text: list, 测试文本，每个元素是一行文本
    '''
    corpus = set()
    num_of_lines = 0
    # 统计总行数
    for line in readFile(filename):
        num_of_lines += 1
    # 生成随机行号并打乱
    all_index = np.arange(num_of_lines)
    np.random.shuffle(all_index)
    # 选择训练集行号
    training_lines = set(all_index[:int(training_ratio * num_of_lines)].tolist())

    testing_text = []
    # 读取文件并构建词库和测试文本
    for index, line in enumerate(readFile(filename)):
        if index not in training_lines:
            # 加入测试文本列表
            testing_text.append(line)
            continue
        for temp in map(lambda x: x.split('/'), line.split('  ')):
            if len(temp) != 2:
                continue
            word, _ = temp
            # 过滤掉包含链接的词语
            if 'ｈｔｔｐ' in word or 'ｗｗｗ．' in word:
                continue
            corpus.add(word)
    return corpus, testing_text

def split_words_reverse(line, corpus_set):
    '''
    反向最大匹配分词

    Parameters
    ----------
    line: str, 需要分词的中文字符串
    corpus_set: set, 词库

    Returns
    ----------
    str: 分词结果
    '''
    n_line = len(line)
    start, end = 0, n_line
    result = []
    while end > 0:
        if (end - 0) == 1:
            # 剩余一个字符时，直接加入结果
            result.append(line[start:end])
            return '/'.join(reversed(result)) + '/'
        current_word = line[start:end]
        if current_word in corpus_set:
            # 如果词语在词库中，加入结果并更新指针
            result.append(current_word)
            end = start
            start = 0
            continue
        else:
            if len(current_word) == 1:
                # 如果是单个字符，将其加入词库并加入结果
                corpus_set.add(current_word)
                result.append(current_word)
                end = start
                start = 0
                continue
            start += 1  # 缩短匹配长度
            continue
        end -= 1
    return '/'.join(reversed(result)) + '/'

def run(split_words_function, testing_text, corpus):
    '''
    运行分词函数并计算分词效果

    Parameters
    ----------
    split_words_function: function, 分词函数，可以是正向或反向匹配
    testing_text: list, 测试文本
    corpus: set, 词库

    Returns
    ----------
    results: list, 分词结果列表
    a, b, c: tuple, 精确度、召回率和 F1 分数的列表
    '''
    p_r_f1 = []
    results = []
    for index, i in enumerate(testing_text):
        text = preprocess_text(i)  # 去掉词性标注
        target = preprocess_text2(i)  # 保留词性标注
        output = split_words_function(text, corpus)  # 分词操作
        p_r_f1.append(precision_recall_f1(output, target))  # 计算精确度、召回率和 F1 分数
        results.append(output)

    # 计算平均精确度、召回率和 F1 分数
    a, b, c = zip(*p_r_f1)
    average_precision = sum(a) / len(a)
    average_recall = sum(b) / len(b)
    average_f1 = sum(c) / len(c)
    print('average precision:', average_precision)
    print('average recall', average_recall)
    print('average F1-score', average_f1)

    # 绘制精确度和召回率的频率直方图
    plt.figure(figsize=(16, 4))
    plt.subplot(121)
    plt.xticks(np.arange(0, 1.05, 0.1))
    plt.hist(a, bins=np.arange(0, 1.05, 0.05))
    plt.xlabel('precision')
    plt.ylabel('frequency')
    plt.title('precision')
    plt.subplot(122)
    plt.xticks(np.arange(0, 1.05, 0.1))
    plt.hist(b, bins=np.arange(0, 1.05, 0.05))
    plt.xlabel('recall')
    plt.ylabel('frequency')
    plt.title('recall')
    plt.show()
    return results, a, b, c

if __name__ == '__main__':
    # 使用人民日报语料库作为词库
    # https://github.com/yaleimeng/NER_corpus_chinese/tree/master
    corpus_file = 'data/PeopleDaily.txt'
    # 构建词库和测试数据
    corpus, testing_text = build_corpus_and_testing_text(corpus_file)
    print('corpus size:', len(corpus))
    print('testing data size:', len(testing_text))
    # 使用反向最大匹配分词并评估结果
    results, a, b, c = run(split_words_reverse, testing_text, corpus)
