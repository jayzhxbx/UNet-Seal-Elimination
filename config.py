import os
import torch

# ===============================  路径  =========================================
# 模型保存文件
if not os.path.exists('./weight'):
    os.mkdir('./weight')

weight = './weight/weight.pth'
weight_with_optimizer = './weight/weight_with_optimizer.pth'
best_model = './weight/best_model.pth'
best_model_with_optimizer = './weight/best_model_with_optimizer.pth'

# ===============================  训练  =========================================
# 选择训练硬件设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 裁剪印章区域后图像的大小
image_size = 512
# 训练参数
n_channels = 1
n_classes = 2
LR = 1e-5
EPOCH = 1000
BATCH_SIZE = 4

# ===============================  测试  =========================================
test_image = './data/train/10.png'
test_boxes = './data/train/10.xml'
output_path = './data/mytest'
