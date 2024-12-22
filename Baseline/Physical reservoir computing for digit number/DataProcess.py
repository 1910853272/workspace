from PIL import Image
import os
import numpy as np

# 定义输入和输出文件夹路径
input_folder = '/Users/lzs/Desktop/workspace/Baseline/Physical reservoir computing for digit number/minist-png'  # 替换为28×28像素图片的文件夹路径
output_folder = '/Users/lzs/Desktop/workspace/Baseline/Physical reservoir computing for digit number/number-png'  # 替换为保存20×20像素图片的文件夹路径
output_data = '/Users/lzs/Desktop/workspace/Baseline/Physical reservoir computing for digit number/data'  # 替换为保存.npy文件的文件夹路径

# 如果输出文件夹不存在，创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if not os.path.exists(output_data):
    os.makedirs(output_data)

# 遍历输入文件夹中的所有图片
for file_name in os.listdir(input_folder):
    # 构建完整路径
    input_path = os.path.join(input_folder, file_name)

    # 检查是否为图片文件
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # 打开图片
        img = Image.open(input_path)

        # 如果图片不是灰度图像，将其转换为灰度图像
        if img.mode != 'L':
            img = img.convert('L')

        # 调整图像大小为20×20像素
        img_resized = img.resize((20, 20), Image.Resampling.LANCZOS)

        # 保存调整后的图像到输出文件夹
        output_path_resized = os.path.join(output_folder, file_name)
        img_resized.save(output_path_resized)

print("批量转换完成，所有20×20像素图片已保存到输出文件夹。")

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

    # 将 20×20 图像展平为 1D 数组
    flattened_img = img_array.flatten()

    # 将展平的数组按每 5 个像素切分，并将其二值化（白色为1，黑色为0）
    binary_blocks = np.array([flattened_img[i:i + 5] > 128 for i in range(0, len(flattened_img), 5)]).astype(int)

    return binary_blocks

# 遍历调整后的文件夹中的所有图片
for file_name in os.listdir(output_folder):  # 遍历已调整为20×20的图片
    # 构建完整路径
    input_path = os.path.join(output_folder, file_name)

    # 检查是否为.png图片
    if file_name.lower().endswith('.png'):
        try:
            # 处理图片并生成二值化矩阵
            binary_matrix = process_image_to_binary_matrix(input_path)

            # 保存二值化矩阵为 .npy 文件
            output_path = os.path.join(output_data, file_name.split('.')[0] + '.npy')
            np.save(output_path, binary_matrix)

            print(f"处理完成：{file_name}，二值化矩阵已保存到 {output_path}")
        except Exception as e:
            print(f"处理图片 {file_name} 时出错：{e}")

print("批量处理完成！")
