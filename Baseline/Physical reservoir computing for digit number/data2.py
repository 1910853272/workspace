import numpy as np
import pandas as pd

# 读取 data.xlsx 文件
df = pd.read_excel('data.xlsx')

# 创建一个空的列表用于存储所有扩充后的数据和标签
all_data = []
all_labels = []

# 每列有80个值，每个值都用正态分布生成40个随机数据，标准差为0.1
num_samples_per_value = 40  # 每个值生成40个数据
std_dev = 0.1  # 标准差

# 遍历每个数字列（digit_0 到 digit_9）
for i in range(10):
    column_name = f'digit_{i}'
    column_values = df[column_name].values  # 获取当前列的80个初始值

    # 创建一个二维数组，用于存储当前数字的生成数据
    # 每个初始值生成40个数据，最终形成40行80列
    digit_data = np.zeros((num_samples_per_value, len(column_values)))

    # 对每个初始值生成40个随机数据
    for j, value in enumerate(column_values):
        digit_data[:, j] = np.random.normal(loc=value, scale=std_dev, size=num_samples_per_value)

    # 创建标签列表，40行对应一个digit
    digit_labels = [column_name] * num_samples_per_value

    # 将生成的数据和标签添加到总体列表中
    all_data.append(digit_data)
    all_labels.extend(digit_labels)

# 将所有数字的数据垂直堆叠，形成一个400行80列的数组
expanded_data_array = np.vstack(all_data)  # shape: (400, 80)

# 将数据转换为DataFrame
# 生成80个特征列的名称
feature_columns = [f'value_{j+1}' for j in range(expanded_data_array.shape[1])]
expanded_df = pd.DataFrame(expanded_data_array, columns=feature_columns)

# 添加标签列到第一列
expanded_df.insert(0, 'label', all_labels)

# 打印部分数据以验证
print(expanded_df.head())

# 保存扩充后的数据为新的 Excel 文件
expanded_df.to_excel('expanded_data_with_labels.xlsx', index=False)

print("数据集已扩充，并保存为 expanded_data_with_labels.xlsx")