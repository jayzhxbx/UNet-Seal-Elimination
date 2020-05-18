import cv2
import numpy as np
import torch
import os
from unet import UNet
from utils import *
import config

# load net
print('load net')
net = UNet(n_channels=config.n_channels, n_classes=config.n_classes).to(config.device)
if os.path.exists(config.best_model):
    checkpoint = torch.load(config.best_model, map_location='cpu')
    net.load_state_dict(checkpoint['net'])
else:
    exit(0)

net.eval()
# load img
print('load img')
image = cv2.imread(config.test_image)
img, _ = cropImage(image, config.test_boxes, image)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img.reshape(img.shape[0], img.shape[1])
print(img.shape)

input = torch.from_numpy(img[np.newaxis][np.newaxis]).float() / 255
output = net(input.to(config.device))
print(output.shape)
output = output[0, 0].detach().data.cpu().numpy()
res = np.concatenate((img / 255, output), axis=1)

cv2.imwrite(os.path.join(config.output_path, "0.png"), res * 255)
