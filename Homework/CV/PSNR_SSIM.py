import numpy as np
import math
import cv2
import os
import pandas as pd

# 自定义PSNR计算函数
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100  # 完全相同
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# 自定义 SSIM 计算函数
def ssim(img1, img2, win_size=9, K1=0.01, K2=0.03, data_range=255):
    """
    计算两张图像的 SSIM（结构相似性）指数。
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # 计算局部均值
    ux = cv2.GaussianBlur(img1, (win_size, win_size), 1.5)
    uy = cv2.GaussianBlur(img2, (win_size, win_size), 1.5)

    # 计算局部方差和协方差
    uxx = cv2.GaussianBlur(img1 * img1, (win_size, win_size), 1.5)
    uyy = cv2.GaussianBlur(img2 * img2, (win_size, win_size), 1.5)
    uxy = cv2.GaussianBlur(img1 * img2, (win_size, win_size), 1.5)

    vx = uxx - ux * ux
    vy = uyy - uy * uy
    vxy = uxy - ux * uy

    # 计算 SSIM
    ssim_map = ((2 * ux * uy + C1) * (2 * vxy + C2)) / ((ux**2 + uy**2 + C1) * (vx + vy + C2))
    return ssim_map.mean()

# 设置文件夹路径
ref_path = 'images/ref_img/'
dist_path = 'images/dist_img/'
output_file = 'score.csv'
image_format = 'png'  # 假设图片为 PNG 格式

# 定义参考图片映射
ref_images = {
    "1600": "1600.png",
    "fisher": "fisher.png",
    "sunset_sparrow": "sunset_sparrow.png"
}

# 初始化列表用于存储结果
image_names = []
psnr_scores = []
ssim_scores = []

# 遍历失真图片文件夹
for dist_img_name in os.listdir(dist_path):
    if dist_img_name.endswith(image_format):
        # 获取失真图片的前缀，查找对应的参考图片
        prefix = dist_img_name.split('.')[0]  # 提取前缀，例如 "1600", "fisher", "sunset_sparrow"
        ref_img_name = ref_images.get(prefix)  # 获取对应的参考图片

        # 检查是否找到了对应的参考图片
        if ref_img_name is None:
            print(f"No matching reference image for {dist_img_name}")
            continue

        # 读取参考图片和失真图片
        ref_img = cv2.imread(os.path.join(ref_path, ref_img_name), cv2.IMREAD_GRAYSCALE)
        dist_img = cv2.imread(os.path.join(dist_path, dist_img_name), cv2.IMREAD_GRAYSCALE)

        if ref_img is not None and dist_img is not None:
            # 计算 PSNR 和 SSIM
            psnr_value = psnr(ref_img, dist_img)
            ssim_value = ssim(ref_img, dist_img)

            # 记录结果
            image_names.append(dist_img_name)
            psnr_scores.append(psnr_value)
            ssim_scores.append(ssim_value)
        else:
            print(f"Could not read {ref_img_name} or {dist_img_name}")

# 创建数据框并保存到 CSV 文件
results_df = pd.DataFrame({
    'ImageName': image_names,
    'PSNR': psnr_scores,
    'SSIM': ssim_scores
})

results_df.to_csv(output_file, index=False, sep=',')
print(f"PSNR and SSIM scores have been saved to {output_file}")
