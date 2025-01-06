import os
import numpy as np
import cv2
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import random

from torchvision.transforms.functional import adjust_brightness

VOC_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair",
               "cow", "diningtable", "dog", "horse", "motorbike",
               "person", "potted plant", "sheep", "sofa", "train",
               "tv/monitor"]

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
                [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
                [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

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
    train_gts = np.zeros(shape=(train_n, 256, 256, 3), dtype=np.uint8)
    test_gts = np.zeros(shape=(test_n, 256, 256, 3), dtype=np.uint8)

    for i in range(train_n):
        train = cv2.imread(os.path.join(path, 'train', f'{train_names[i]}.jpg'))
        train_gt = cv2.imread(os.path.join(path, 'train_gt', f'{train_names[i]}.png'))
        train = cv2.resize(train, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
        train_gt = cv2.resize(train_gt, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        # cv2.imshow('train', train)
        # cv2.imshow('train_gt', train_gt)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        trains[i, :, :, :] = train
        train_gts[i, :, :, :] = train_gt
        # if i % 500 == 0: print(f'{i}/{train_n}')
    print(f'train data load finished')

    for i in range(test_n):
        test = cv2.imread(os.path.join(path, 'test', f'{test_names[i]}.jpg'))
        test_gt = cv2.imread(os.path.join(path, 'test_gt', f'{test_names[i]}.png'))
        test = cv2.resize(test, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
        test_gt = cv2.resize(test_gt, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        # cv2.imshow('test', test)
        # cv2.imshow('test_gt', test_gt)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        tests[i, :, :, :] = test
        test_gts[i, :, :, :] = test_gt
        # if i % 500 == 0: print(f'{i}/{test_n}')
    print(f'test data load finished')

    return trains, tests, train_gts, test_gts

def mini_batch(data_size, batch_size, img, gt):
    idx = np.random.randint(low=0, high=data_size, size=batch_size)
    batch_img = np.zeros(shape=(batch_size, 256, 256, 3), dtype=np.float32)
    batch_gt = np.zeros(shape=(batch_size, 256, 256), dtype=np.int64)

    for i in range(batch_size):
        aug_img = img[idx[i], :, :, :]
        aug_gt = gt[idx[i], :, :, :]

        p = [round(random.random(), 1) for j in range(7)]   # 0~1 float

        aug_img = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV)

        if p[0] < 0.8:
            aug_img[:, :, 0] *= random.choice([0.5, 1.5])
        if p[1] < 0.8:
            aug_img[:, :, 1] *= random.choice([0.5, 1.5])
        if p[2] < 0.8:
            aug_img[:, :, 2] *= random.choice([0.7, 1.05])
        if p[3] < 0.8:
            aug_img = cv2.GaussianBlur(aug_img, (5, 5), 2)

        aug_img = cv2.cvtColor(aug_img, cv2.COLOR_HSV2RGB)
        aug_gt = cv2.cvtColor(aug_gt, cv2.COLOR_BGR2RGB)

        if p[4] < 0.8:
            ang = random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((aug_img.shape[1] / 2, aug_img.shape[0] / 2), ang, 1)
            aug_img = cv2.warpAffine(aug_img, M, (aug_img.shape[1], aug_img.shape[0]))
            aug_gt = cv2.warpAffine(aug_gt, M, (aug_img.shape[1], aug_img.shape[0]))
        if p[5] < 0.8:
            aug_img = cv2.flip(aug_img, 0)
            aug_gt = cv2.flip(aug_gt, 0)
        if p[6] < 0.8:
            aug_img = cv2.flip(aug_img, 1)
            aug_gt = cv2.flip(aug_gt, 1)

        batch_img[i, :, :, :] = aug_img / 255.0 * 2 - 1
        batch_gt[i, :, :] = color2index(aug_gt)

    batch_img = np.transpose(batch_img, (0, 3, 1, 2))

    return batch_img, batch_gt


def color2index(img):
    image = np.zeros(shape=(256, 256), dtype=np.uint8)
    for index, color in enumerate(VOC_COLORMAP):
        where = np.all(img == color, axis=2)
        image[where] = index
    return image


def index2color(img):
    output = np.zeros(shape=(256, 256, 3), dtype=np.uint8)
    for p in range(256):
        for q in range(256):
            output[p, q] = VOC_COLORMAP[img[p, q]]
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output

def augmentation(img, gt):
    p = 0
    if p == 0:
        img = TF.adjust_brightness(img=img, brightness_factor=0.9)
    elif p == 1:
        img = TF.adjust_hue(img=img, hue_factor=0.9)
    elif p == 2:
        img = TF.adjust_saturation(img=img, saturation_factor=1.01)
    elif p == 3:
        img = TF.adjust_sharpness(img=img, sharpness_factor=2)

    p = 1

    if p < 0.3:
        ang = random.randrange(-10, 10)
        img = TF.rotate(img=img, angle=ang)
        gt = TF.rotate(img=gt, angle=ang)
    # p = random.random()

    if p < 0.3:
        img = TF.vflip(img=img)
        gt = TF.vflip(img=gt)
    # p = random.random()

    if p < 0.3:
        img = TF.hflip(img=img)
        gt = TF.hflip(img=gt)

    return img, gt
