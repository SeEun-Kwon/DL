import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import torchvision
from PIL import Image

VOC_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
               "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
               "potted plant", "sheep", "sofa", "train", "tv/monitor"]

VOC_COLORMAP = np.array([[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128], [128, 0, 0], [128, 0, 128],
                [128, 128, 0], [128, 128, 128], [0, 0, 64], [0, 0, 192], [0, 128, 64], [0, 128, 192],
                [128, 0, 64], [128, 0, 192], [128, 128, 64], [128, 128, 192], [0, 64, 0], [0, 64, 128],
                [0, 192, 0], [0, 192, 128], [128, 64, 0]])
# bgr

def data_load(path):
    train_names = open(path+'/ImageSets/Segmentation/train.txt', 'r').readlines()
    test_names = open(path+'/ImageSets/Segmentation/val.txt', 'r').readlines()
    for i, name in enumerate(train_names):
        train_names[i] = name.split('\n')[0]
    for i, name in enumerate(test_names):
        test_names[i] = name.split('\n')[0]

    train_n = len(train_names)
    test_n = len(test_names)

    trains = np.zeros(shape=(train_n, 256, 256, 3), dtype=np.float32)
    tests = np.zeros(shape=(test_n, 256, 256, 3), dtype=np.float32)
    train_gts = np.zeros(shape=(train_n, 256, 256), dtype=np.uint8)
    test_gts = np.zeros(shape=(test_n, 256, 256), dtype=np.uint8)

    for i in range(train_n):
        train = cv2.imread(path + '/train/' + train_names[i] + '.jpg')
        train_gt = cv2.imread(path + '/train_gt/' + train_names[i] + '.png')
        train = cv2.resize(train, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
        train_gt = cv2.resize(train_gt, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        trains[i, :, :, :] = train
        train_gts[i, :, :] = color_to_index(train_gt)
        if i % 500 == 0: print(f'{i}/{train_n}')
    print(f'original train data load finished')

    for i in range(test_n):
        test = cv2.imread(path + '/test/' + test_names[i] + '.jpg')
        test_gt = cv2.imread(path + '/test_gt/' + test_names[i] + '.png')
        test = cv2.resize(test, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
        test_gt = cv2.resize(test_gt, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        tests[i, :, :, :] = test
        test_gts[i, :, :] = color_to_index(test_gt)
        if i % 500 == 0: print(f'{i}/{test_n}')
    print(f'test data load finished')

    return trains, tests, train_gts, test_gts

def mini_batch(data_size, batch_size, img, gt):
    idx = np.random.randint(low=0, high=data_size, size=batch_size)
    batch_img = np.zeros(shape=(batch_size, 256, 256, 3), dtype=np.float32)
    batch_gt = np.zeros(shape=(batch_size, 256, 256), dtype=np.uint8)


    for i in range(batch_size):
        p = np.random.randint(0, 2, 1)

        aug_img = img[idx[i], :, :, :]
        aug_gtimg = gt[idx[i], :, :]
        if p:
            aug_img = (cv2.flip(aug_img, 0))
            aug_gtimg = cv2.flip(aug_gtimg, 0)
        p = np.random.randint(0, 2, 1)
        if p:
            M = cv2.getRotationMatrix2D((256/2, 256/2), 15, 1)
            aug_img = cv2.warpAffine(aug_img, M, (256, 256))
            aug_gtimg = cv2.warpAffine(aug_gtimg, M, (256, 256))
        # p = np.random.randint(0, 2, 1)
        # if p:
        #     img = Image.fromarray(img)
        #     img = torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)(img)
        #     img = np.array(img)
        batch_img[i, :, :, :] = aug_img / 255.0 * 2 - 1.0
        batch_gt[i, :, :] = aug_gtimg

    batch_img = np.transpose(batch_img, (0, 3, 1, 2))

    return batch_img, batch_gt


def color_to_index(img):
    image = np.zeros(shape=(256, 256), dtype=np.uint8)
    for index, color in enumerate(VOC_COLORMAP):
        where = np.all(img == color, axis=2)
        image[where] = index
    return image


def index_to_color(img):
    output = np.zeros(shape=(256, 256, 3), dtype=np.uint8)
    for p in range(256):
        for q in range(256):
            output[p, q] = VOC_COLORMAP[img[p, q]]
    return output
