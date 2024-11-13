#!/bin/bash

# 运行 PSNR_SSIM.py 脚本并将输出结果保存为 score_test.csv
python3 PSNR_SSIM.py

# 将生成的 score.csv 文件重命名为 score_test.csv
mv score.csv score_test.csv

echo "PSNR and SSIM scores have been saved to score_test.csv"
