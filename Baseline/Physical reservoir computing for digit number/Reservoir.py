import pandas as pd
import numpy as np
from IPython.display import display

# 从Excel文件中加载数据到DataFrame
data = pd.read_excel("./data.xlsx")

# 将电导值从西门子 (S) 转换为纳西门子 (nS)
data *= 1E9

# 设置标准差为 0.1
Standard_deviation = 0.1

# 创建一个空的 DataFrame 用于存储训练数据
train_data_01 = pd.DataFrame()

# 使用正态分布的随机数据模拟实验或观测数据的随机性

for i in range(0, 5):
    # 创建 'LETTER' 列，生成服从正态分布的随机数据，均值为 i，标准差为 0
    data0 = pd.DataFrame({'LETTER': np.random.normal(i, 0, size=100)})

    # 生成 'CELL1' 列，数据的均值是 data 中第 5*i 行的最大值和最小值的均值，标准差为 0.1
    data1 = pd.DataFrame({'CELL1': np.random.normal(np.mean([np.max(data.iloc[5*i+0]), np.min(data.iloc[5*i+0])]), 0.1, size=100)})

    # 生成 'CELL2' 列，数据的均值是 data 中第 5*i+1 行的最大值和最小值的均值，标准差为 0.1
    data2 = pd.DataFrame({'CELL2': np.random.normal(np.mean([np.max(data.iloc[5*i+1]), np.min(data.iloc[5*i+1])]), 0.1, size=100)})

    # 生成 'CELL3' 列，数据的均值是 data 中第 5*i+2 行的最大值和最小值的均值，标准差为 0.1
    data3 = pd.DataFrame({'CELL3': np.random.normal(np.mean([np.max(data.iloc[5*i+2]), np.min(data.iloc[5*i+2])]), 0.1, size=100)})

    # 生成 'CELL4' 列，数据的均值是 data 中第 5*i+3 行的最大值和最小值的均值，标准差为 0.1
    data4 = pd.DataFrame({'CELL4': np.random.normal(np.mean([np.max(data.iloc[5*i+3]), np.min(data.iloc[5*i+3])]), 0.1, size=100)})

    # 生成 'CELL5' 列，数据的均值是 data 中第 5*i+4 行的最大值和最小值的均值，标准差为 0.1
    data5 = pd.DataFrame({'CELL5': np.random.normal(np.mean([np.max(data.iloc[5*i+4]), np.min(data.iloc[5*i+4])]), 0.1, size=100)})

    # 将 data0, data1, data2, data3, data4, data5 沿列方向合并，然后将其添加到训练数据集中
    train_data_01 = pd.concat([train_data_01, pd.concat([data0, data1, data2, data3, data4, data5], axis=1)])

# 显示生成的训练数据
display(train_data_01)

# 将训练数据保存为 CSV 文件，编码方式为 CP949，且不保存索引
train_data_01.to_csv("train_data_01.csv", encoding="CP949", index=False)

# 创建一个空的 DataFrame 用于存储所有数据
total_data = pd.DataFrame()

# 循环处理五组数据
for i in range(0, 5):
    # 选取 train_data_01 中的第 i 组数据（100 行数据）
    total = train_data_01[100*i:100*(i+1)]

    # 将选定的数据从宽格式转换为长格式，只保留 'CELL1', 'CELL2', 'CELL3', 'CELL4', 'CELL5' 列
    total = pd.melt(total, value_vars=['CELL1', 'CELL2', 'CELL3', 'CELL4', 'CELL5'])

    # 将每次处理的数据连接到 total_data 中，沿列方向合并
    total_data = pd.concat([total_data, total], axis=1)

# 显示合并后的总数据
display(total_data)

# 设置标准差为 0.1
Standard_deviation = 0.1

# 创建一个空的 DataFrame 用于存储测试数据
test_data_01 = pd.DataFrame()

# 循环生成五组数据
for i in range(0, 5):
    # 创建 'LETTER' 列，生成服从正态分布的随机数据，均值为 i，标准差为 0
    data0 = pd.DataFrame({'LETTER': np.random.normal(i, 0, size=100)})

    # 生成 'CELL1' 列，数据的均值是 data 中第 5*i 行的最大值和最小值的均值，标准差为 0.1
    data1 = pd.DataFrame({'CELL1': np.random.normal(np.mean([np.max(data.iloc[5*i+0]), np.min(data.iloc[5*i+0])]), 0.1, size=100)})

    # 生成 'CELL2' 列，数据的均值是 data 中第 5*i+1 行的最大值和最小值的均值，标准差为 0.1
    data2 = pd.DataFrame({'CELL2': np.random.normal(np.mean([np.max(data.iloc[5*i+1]), np.min(data.iloc[5*i+1])]), 0.1, size=100)})

    # 生成 'CELL3' 列，数据的均值是 data 中第 5*i+2 行的最大值和最小值的均值，标准差为 0.1
    data3 = pd.DataFrame({'CELL3': np.random.normal(np.mean([np.max(data.iloc[5*i+2]), np.min(data.iloc[5*i+2])]), 0.1, size=100)})

    # 生成 'CELL4' 列，数据的均值是 data 中第 5*i+3 行的最大值和最小值的均值，标准差为 0.1
    data4 = pd.DataFrame({'CELL4': np.random.normal(np.mean([np.max(data.iloc[5*i+3]), np.min(data.iloc[5*i+3])]), 0.1, size=100)})

    # 生成 'CELL5' 列，数据的均值是 data 中第 5*i+4 行的最大值和最小值的均值，标准差为 0.1
    data5 = pd.DataFrame({'CELL5': np.random.normal(np.mean([np.max(data.iloc[5*i+4]), np.min(data.iloc[5*i+4])]), 0.1, size=100)})

    # 将 data0, data1, data2, data3, data4, data5 沿列方向合并，然后将其添加到测试数据集中
    test_data_01 = pd.concat([test_data_01, pd.concat([data0, data1, data2, data3, data4, data5], axis=1)])

# 显示生成的测试数据
display(test_data_01)

# 将测试数据保存为 CSV 文件，编码方式为 CP949，且不保存索引
test_data_01.to_csv("test_data_01.csv", encoding="CP949", index=False)

# 创建一个空的 DataFrame 用于存储所有数据
total_data2 = pd.DataFrame()

# 循环处理十五组数据
for i in range(0, 15):
    # 选取 test_data_01 中的第 i 组数据（100 行数据）
    total2 = test_data_01[100*i:100*(i+1)]

    # 将选定的数据从宽格式转换为长格式，只保留 'CELL1', 'CELL2', 'CELL3', 'CELL4', 'CELL5' 列
    total2 = pd.melt(total2, value_vars=['CELL1', 'CELL2', 'CELL3', 'CELL4', 'CELL5'])

    # 将每次处理的数据连接到 total_data2 中，沿列方向合并
    total_data2 = pd.concat([total_data2, total2], axis=1)

# 显示合并后的总数据
display(total_data2)


