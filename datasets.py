from torch.utils.data import Dataset
import cv2
import os
import transforms as Transforms
from utils import *


class UNetDataset(Dataset):
    def __init__(self, dir_train, dir_mask, transform=None):
        self.dirTrain = dir_train
        self.dirMask = dir_mask
        self.transform = transform
        self.dataTrain = [os.path.join(self.dirTrain, filename)
                          for filename in os.listdir(self.dirTrain)
                          if filename.endswith('.jpg') or filename.endswith('.png')]
        self.dataBox = [os.path.join(self.dirTrain, filename)
                        for filename in os.listdir(self.dirTrain)
                        if filename.endswith('.xml')]
        self.dataMask = [os.path.join(self.dirMask, filename)
                         for filename in os.listdir(self.dirMask)
                         if filename.endswith('.jpg') or filename.endswith('.png')]
        self.trainDataSize = len(self.dataTrain)
        self.maskDataSize = len(self.dataMask)
        self.dataBoxSize = len(self.dataBox)

    def __getitem__(self, index):
        assert self.trainDataSize == self.maskDataSize
        assert self.trainDataSize == self.dataBoxSize

        image = cv2.imread(self.dataTrain[index])
        label = cv2.imread(self.dataMask[index])
        boxfile = self.dataBox[index]

        image, label = cropImage(image, boxfile, label)

        if self.transform:
            for method in self.transform:
                image, label = method(image, label)

        image = image.transpose((2, 0, 1))
        label = label.transpose((2, 0, 1))

        return image, label

    def __len__(self):
        assert self.trainDataSize == self.maskDataSize
        assert self.trainDataSize == self.dataBoxSize

        return self.trainDataSize


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    transforms = [
        # Transforms.RandomCrop(2300, 2300),
        Transforms.RondomFlip(),
        Transforms.RandomRotate(15),
        Transforms.Log(0.5),
        Transforms.Blur(0.2),
        Transforms.ToTensor(),
        Transforms.ToGray()
    ]
    dataset = UNetDataset('./data/train', './data/train_cleaned', transform=transforms)
    dataLoader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)

    for index, (batch_x, batch_y) in enumerate(dataLoader):
        print(batch_x.size(), batch_y.size())  # shape:(batch_size, 1, h, w)  1表示图像是灰度图

        dis = batch_y[0][0].numpy()  # shape:(h,w)
