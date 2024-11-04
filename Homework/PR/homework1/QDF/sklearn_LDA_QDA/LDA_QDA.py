# 使用pip install -U scikit-learn 安装sklearn库
# 使用pip install numpy安装numpy库（可能内置）

# 引入必要的库
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA

# 函数：用于从 letter 数据集中提取类别和特征
def get_list_letter(char, features, path):
    # 打开数据集文件并读取所有行
    with open(path, 'r') as letter_train:
        content_letter_train = letter_train.readlines()
    for line in content_letter_train:
        # 使用逗号分割每一行数据
        temp = line.split(',')
        temp[-1] = list(temp[-1])[0]  # 获取类别，例如字母'A'
        char.append(temp[0])  # 将类别（如字母）追加到 char 列表中
        features.append(temp[1::])  # 将特征数据追加到 features 列表中

# 函数：用于从 optdigits 数据集中提取类别和特征
def get_list_optdigits(dig, features, path):
    with open(path, 'r') as dig_train:
        content_dig_train = dig_train.readlines()
    for line in content_dig_train:
        temp = line.split(',')
        temp[-1] = list(temp[-1])[0]  # 获取类别（例如数字0-9）
        dig.append(temp[-1])  # 将类别（数字）追加到 dig 列表中
        features.append(temp[0:len(temp)-1:])  # 将特征数据追加到 features 列表中

# 函数：用于从 sat 数据集中提取类别和特征
def get_list_sat(sat, features, path):
    with open(path, 'r') as sat_train:
        content_sat_train = sat_train.readlines()
    for line in content_sat_train:
        temp = line.split(' ')
        temp[-1] = list(temp[-1])[0]  # 获取类别
        sat.append(temp[-1])  # 将类别追加到 sat 列表中
        features.append(temp[0:len(temp)-1:])  # 将特征数据追加到 features 列表中

# 函数：用于从 vowel 数据集中提取类别和特征
def get_list_vowel(vowel, features, path):
    with open(path, 'r') as vowel_train:
        content_vowel_train = vowel_train.readlines()
    for line in content_vowel_train:
        temp = line.split()
        temp[-1] = list(temp[-1])[0]  # 获取类别
        vowel.append(temp[-1])  # 将类别追加到 vowel 列表中
        features.append(temp[3:len(temp)-1:])  # 将特征数据追加到 features 列表中

# 函数：进行训练并测试数据，返回分类结果
def train_and_test(train_result, fetures_list, test_list, analysis_method):
    # 将训练数据转换为 numpy 数组
    letter_x = np.array(fetures_list)
    letter_y = np.array(train_result)
    # 选择 LDA 或 QDA 作为分析方法
    if analysis_method == "LDA":
        clf = LDA()
    elif analysis_method == "QDA":
        clf = QDA()
    else:
        print("错误的分类方法名。")
        return -1
    # 训练模型
    clf.fit(letter_x, letter_y)
    # 返回测试数据的预测结果
    return list(clf.predict(test_list))

# 函数：将特征数据从字符串转换为整数
def convert_int(str_list):
    for row in range(0, len(str_list)):
        for col in range(0, len(str_list[row])):
            str_list[row][col] = int(str_list[row][col])

# 函数：将特征数据从字符串转换为浮点数
def convert_float(str_list):
    for row in range(0, len(str_list)):
        for col in range(0, len(str_list[row])):
            str_list[row][col] = float(str_list[row][col])

# 函数：计算预测结果的分类准确率
def analysis_accuracy(judge_result, test_char):
    sum = 0  # 总样本数
    right_num = 0  # 预测正确的样本数
    for pos in range(0, len(judge_result)):
        sum += 1
        if judge_result[pos] == test_char[pos]:
            right_num += 1
    return right_num / sum  # 返回准确率

# letter 数据集初始化
letter_train_path = './dataset/letter.train'  # 训练集文件路径
letter_train_class = []  # 用于保存训练集类别
letter_train_features = []  # 用于保存训练集特征
letter_test_path = './dataset/letter.test'  # 测试集文件路径
letter_test_class = []  # 用于保存测试集类别
letter_test_features = []  # 用于保存测试集特征

# 加载训练集和测试集
get_list_letter(letter_train_class, letter_train_features, letter_train_path)
get_list_letter(letter_test_class, letter_test_features, letter_test_path)
convert_int(letter_train_features)
convert_int(letter_test_features)

# 使用 LDA 对 letter 数据集进行训练和测试
letter_LDA_judge_result = train_and_test(letter_train_class, letter_train_features, letter_test_features, "LDA")
letter_LDA_judge_accuracy = analysis_accuracy(letter_LDA_judge_result, letter_test_class)
print('使用LDA对letter的', len(letter_train_features), '份数据学习后，对',
      len(letter_test_features), '份测试数据分类的准确率为：', letter_LDA_judge_accuracy)

# 使用 QDA 对 letter 数据集进行训练和测试
letter_QDA_judge_result = train_and_test(letter_train_class, letter_train_features, letter_test_features, "QDA")
letter_QDA_judge_accuracy = analysis_accuracy(letter_QDA_judge_result, letter_test_class)
print('使用QDA对letter的', len(letter_train_features), '份数据学习后，对',
      len(letter_test_features), '份测试数据分类的准确率为：', letter_QDA_judge_accuracy)

# optdigits 数据集初始化
optdigits_train_path = './dataset/optdigits.train'  # 训练集文件路径
optdigits_train_class = []  # 用于保存训练集类别
optdigits_train_features = []  # 用于保存训练集特征
optdigits_test_path = './dataset/optdigits.test'  # 测试集文件路径
optdigits_test_class = []  # 用于保存测试集类别
optdigits_test_features = []  # 用于保存测试集特征

# 加载训练集和测试集
get_list_optdigits(optdigits_train_class, optdigits_train_features, optdigits_train_path)
convert_int(optdigits_train_features)
get_list_optdigits(optdigits_test_class, optdigits_test_features, optdigits_test_path)
convert_int(optdigits_test_features)

# 使用 LDA 对 optdigits 数据集进行训练和测试
optdigits_LDA_judge_result = train_and_test(optdigits_train_class, optdigits_train_features, optdigits_test_features, "LDA")
optdigits_LDA_judge_accuracy = analysis_accuracy(optdigits_LDA_judge_result, optdigits_test_class)
print('使用LDA对optdigits的', len(optdigits_train_features), '份数据学习后，对',
      len(optdigits_test_features), '份测试数据分类的准确率为：', optdigits_LDA_judge_accuracy)

# 使用 QDA 对 optdigits 数据集进行训练和测试
optdigits_QDA_judge_result = train_and_test(optdigits_train_class, optdigits_train_features, optdigits_test_features, "QDA")
optdigits_QDA_judge_accuracy = analysis_accuracy(optdigits_QDA_judge_result, optdigits_test_class)
print('使用QDA对optdigits的', len(optdigits_train_features), '份数据学习后，对',
      len(optdigits_test_features), '份测试数据分类的准确率为：', optdigits_QDA_judge_accuracy)

# sat 数据集初始化
sat_train_path = './dataset/sat.train'  # 训练集文件路径
sat_train_class = []  # 用于保存训练集类别
sat_train_features = []  # 用于保存训练集特征
sat_test_path = './dataset/sat.test'  # 测试集文件路径
sat_test_class = []  # 用于保存测试集类别
sat_test_features = []  # 用于保存测试集特征

# 加载训练集和测试集
get_list_sat(sat_train_class, sat_train_features, sat_train_path)
convert_int(sat_train_features)
get_list_sat(sat_test_class, sat_test_features, sat_test_path)
convert_int(sat_test_features)

# 使用 LDA 对 sat 数据集进行训练和测试
sat_LDA_judge_result = train_and_test(sat_train_class, sat_train_features, sat_test_features, "LDA")
sat_LDA_judge_accuracy = analysis_accuracy(sat_LDA_judge_result, sat_test_class)
print('使用LDA对sat的', len(sat_train_features), '份数据学习后，对',
      len(sat_test_features), '份测试数据分类的准确率为：', sat_LDA_judge_accuracy)

# 使用 QDA 对 sat 数据集进行训练和测试
sat_QDA_judge_result = train_and_test(sat_train_class, sat_train_features, sat_test_features, "QDA")
sat_QDA_judge_accuracy = analysis_accuracy(sat_QDA_judge_result, sat_test_class)
print('使用QDA对sat的', len(sat_train_features), '份数据学习后，对',
      len(sat_test_features), '份测试数据分类的准确率为：', sat_QDA_judge_accuracy)

# vowel 数据集初始化
vowel_train_path = './dataset/vowel.train'  # 训练集文件路径
vowel_train_class = []  # 用于保存训练集类别
vowel_train_features = []  # 用于保存训练集特征
vowel_test_path = './dataset/vowel.test'  # 测试集文件路径
vowel_test_class = []  # 用于保存测试集类别
vowel_test_features = []  # 用于保存测试集特征

# 加载训练集和测试集
get_list_vowel(vowel_train_class, vowel_train_features, vowel_train_path)
convert_float(vowel_train_features)
get_list_vowel(vowel_test_class, vowel_test_features, vowel_test_path)
convert_float(vowel_test_features)

# 使用 LDA 对 vowel 数据集进行训练和测试
vowel_LDA_judge_result = train_and_test(vowel_train_class, vowel_train_features, vowel_test_features, "LDA")
vowel_LDA_judge_accuracy = analysis_accuracy(vowel_LDA_judge_result, vowel_test_class)
print('使用LDA对vowel的', len(vowel_train_features), '份数据学习后，对',
      len(vowel_test_features), '份测试数据分类的准确率为：', vowel_LDA_judge_accuracy)

# 使用 QDA 对 vowel 数据集进行训练和测试
vowel_QDA_judge_result = train_and_test(vowel_train_class, vowel_train_features, vowel_test_features, "QDA")
vowel_QDA_judge_accuracy = analysis_accuracy(vowel_QDA_judge_result, vowel_test_class)
print('使用QDA对vowel的', len(vowel_train_features), '份数据学习后，对',
      len(vowel_test_features), '份测试数据分类的准确率为：', vowel_QDA_judge_accuracy)
