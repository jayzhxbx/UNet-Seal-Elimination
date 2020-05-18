import os
import random
import xml.etree.ElementTree as ET
import numpy.random as npr
import cv2
import config

image_size = config.image_size


def getBoxes(xml_file):
    boxes = []
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for member in root.findall('object'):
        box = [int(member[4][0].text),
               int(float(member[4][1].text)),
               int(member[4][2].text),
               int(member[4][3].text)
               ]
        boxes.append(box)
    return boxes


def getImage(image, label, boxes):
    cropCode = random.choice([-1] + [i for i in range(len(boxes))])
    height, width = image.shape[0], image.shape[1]
    if cropCode != -1:
        box = boxes[cropCode]
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        w = xmax - xmin
        h = ymax - ymin
        if max(w, h) < image_size:
            nx = npr.randint(max((xmax - image_size), 0), xmin)
            ny = npr.randint(max((ymax - image_size), 0), ymin)
            if nx + image_size > width:
                nx = width - image_size
            if ny + image_size > height:
                ny = height - image_size
            cropped_im = image[ny: ny + image_size, nx: nx + image_size, :]
            cropped_la = label[ny: ny + image_size, nx: nx + image_size, :]
        else:
            nx = npr.randint(xmax - max(w, h) - 100, xmin)
            ny = npr.randint(ymax - max(w, h) - 100, ymin)
            if nx + max(w, h) + 100 > width:
                nx = width - max(w, h) - 100
            if ny + max(w, h) + 100 > height:
                ny = height - max(w, h) - 100
            cropped_im = image[ny: ny + max(w, h) + 100, nx: nx + max(w, h) + 100, :]
            cropped_la = label[ny: ny + max(w, h) + 100, nx: nx + max(w, h) + 100, :]
    else:
        crop_x = int(random.random() * (width - image_size))
        crop_y = int(random.random() * (height - image_size))
        cropped_im = image[crop_y: crop_y + image_size, crop_x: crop_x + image_size]
        cropped_la = label[crop_y: crop_y + image_size, crop_x: crop_x + image_size]

    return cropped_im, cropped_la


def cropImage(image, boxfile, label):
    boxes = getBoxes(boxfile)
    image, label = getImage(image, label, boxes)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return image, label


if __name__ == '__main__':
    dirTrain = './data/train'
    dataBox = [os.path.join(dirTrain, filename)
               for filename in os.listdir(dirTrain)
               if filename.endswith('.xml')]
    dataTrain = [os.path.join(dirTrain, filename)
                 for filename in os.listdir(dirTrain)
                 if filename.endswith('.jpg') or filename.endswith('.png')]

    for i in range(len(dataTrain)):
        image = cv2.imread(dataTrain[i])
        boxes = getBoxes(dataBox[i])
        crop_image, cropped_label = getImage(image, boxes)
        cv2.imwrite('./test/{}.png'.format(i), crop_image)
