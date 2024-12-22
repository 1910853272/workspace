import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from collections import Counter

# 输入信号和对应的输出值映射
data_mapping = {
    '0000': 0,
    '0001': 85.89,
    '0010': 72.72,
    '0011': 132.42,
    '0100': 51.78,
    '0101': 126.98,
    '0110': 101.31,
    '0111': 219.68,
    '1000': 35.78,
    '1001': 123.67,
    '1010': 93.48,
    '1011': 193.32,
    '1100': 81.43,
    '1101': 214.48,
    '1110': 164.17,
    '1111': 251.4
}

def output_row(initial_state, input_signal):
    """
    模拟忆阻器行为，基于数据映射返回输出值。
    """
    # 确保 input_signal 是 NumPy 数组
    input_signal = np.array(input_signal)

    # 将输入信号转为四位二进制字符串
    signal_str = ''.join([str(int(x)) for x in input_signal.flatten()])  # 使用 flatten() 确保是1D数组

    # 打印调试信息：输出转换后的 signal_str
    print(f"Converted input signal: {input_signal} -> {signal_str}")

    # 从映射中获取输出值
    output = data_mapping.get(signal_str, None)

    if output is None:
        print(f"Warning: 输入信号 {signal_str} 在映射中找不到。")
        # 打印所有映射的键，以便调试
        print(f"Available mapping keys: {list(data_mapping.keys())}")
        return None

    return output


