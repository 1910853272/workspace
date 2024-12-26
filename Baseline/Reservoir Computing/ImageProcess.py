from PIL import Image
import os
import numpy as np

# 定义输入和输出文件夹路径
input_folder = 'C:\\Users\\19108\\Desktop\\workspace\\Baseline\\Physical reservoir computing for digit number\\minist-png'  # 替换为28×28像素图片的文件夹路径
output_folder = 'C:\\Users\\19108\\Desktop\\workspace\\Baseline\\Physical reservoir computing for digit number\\number-png'  # 替换为保存20×20像素图片的文件夹路径
output_data = 'C:\\Users\\19108\\Desktop\\workspace\\Baseline\\Physical reservoir computing for digit number\\data'  # 保存 .npy 文件的文件夹

# 如果输出文件夹不存在，创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if not os.path.exists(output_data):
    os.makedirs(output_data)

# 批量处理函数
def process_image_to_binary_matrix(image_path):
    # 打开图片
    img = Image.open(image_path)

    # 将图片转换为灰度图像（如果不是灰度图像）
    if img.mode != 'L':
        img = img.convert('L')

    # 调整图像大小为 20×20 像素（确保尺寸正确）
    img = img.resize((20, 20), Image.Resampling.LANCZOS)

    # 转换为 numpy 数组
    img_array = np.array(img)

    # 二值化处理：将像素值大于128的设置为1，其他设置为0
    binary_matrix = (img_array > 128).astype(int)

    # 返回二值化的图像矩阵（每行为一维数组）
    return binary_matrix

# 只处理 digit0 到 digit9 的 10 张图片
for i in range(10):
    digit_label = f'digit_{i}'  # 构造数字标签
    # 构造图像文件的路径
    image_path = os.path.join(input_folder, f'{digit_label}.png')

    # 检查文件是否存在
    if os.path.exists(image_path):
        # 处理图像并获取二值化矩阵
        binary_matrix = process_image_to_binary_matrix(image_path)

        # 构造输出的.npy文件路径
        output_file_path = os.path.join(output_data, f'{digit_label}.npy')

        # 将二值化矩阵保存为.npy文件
        np.save(output_file_path, binary_matrix)

        # 打印保存的.npy文件中的值
        loaded_matrix = np.load(output_file_path)
        print(f'已处理并保存 {digit_label} -> {output_file_path}')
        print(f'保存的矩阵内容：\n{loaded_matrix}\n')
    else:
        print(f'文件不存在: {image_path}')

