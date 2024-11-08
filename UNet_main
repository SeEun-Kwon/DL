import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from network import UNet
from UNet_function import data_load, mini_batch, index_to_color
import time
from tqdm import tqdm
import os
import cv2
from mIoU_git import *

# data load
print('Datasets loading...')
path = '/home/aivs/바탕화면/hdd/dataset/VOC2012'
try_name = '11.8'

train, test, train_gts, test_gts = data_load(path)

VOC_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
               "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
               "potted plant", "sheep", "sofa", "train", "tv/monitor"]

VOC_COLORMAP = np.array([[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128], [128, 0, 0], [128, 0, 128],
                [128, 128, 0], [128, 128, 128], [0, 0, 64], [0, 0, 192], [0, 128, 64], [0, 128, 192],
                [128, 0, 64], [128, 0, 192], [128, 128, 64], [128, 128, 192], [0, 64, 0], [0, 64, 128],
                [0, 192, 0], [0, 192, 128], [128, 64, 0]])

test_names = open(path+'/ImageSets/Segmentation/val.txt', 'r').readlines()
for i in range(len(test_names)):
    test_names[i] = test_names[i].strip('\n')
train_n, test_n, class_num = len(train), len(test), len(VOC_CLASSES)

batch_size = 32
lr = 0.0001
iter = 200000
start =  10000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
model_load = 10000
model.load_state_dict(torch.load(f'model/{try_name}_{model_load}.pt'))
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
start_time = time.time()


print('train start')

# train # tensorbord에 결과 저장 가능
for i in range(start, iter + 1):
    if i == 30000:
        optimizer.param_groups[0]['lr'] /= 0.1

    f1 = open(f'UNet loss_{try_name}.txt', 'a+')
    f2 = open(f'UNet accuracy_{try_name}.txt', 'a+')
    model.train()

    optimizer.zero_grad()
    batch_img, batch_gts = mini_batch(train_n, batch_size, train, train_gts)
    batch_img, batch_gts = torch.from_numpy(batch_img), torch.from_numpy(batch_gts).type(torch.LongTensor)
    batch_img, batch_gts = batch_img.to(device), batch_gts.to(device)

    output = model(batch_img)
    loss = criterion(output, batch_gts)
    loss.backward()
    optimizer.step()
    if i % 500 == 0:
        print(f'{i}\t{loss.item():.6f}')
        f1.write(f'{i}\t{loss.item():.6f}\n')

    # test
    if i % 5000 == 0:
        torch.save(model.state_dict(), f'model/{try_name}_{i}.pt')
        model.eval()

        #initialize
        total_mIoU, total_pixel_accuracy = 0., 0.
        total_c_m = np.zeros(shape=(21, 21), dtype=np.uint32)

        with (torch.no_grad()):
            for j in tqdm(range(test_n), desc='testing...'):
                test_img = test[j:j+1, :, :, :] / 255.0 * 2.0 - 1.0
                test_img = np.transpose(test_img, (0, 3, 1, 2))
                test_img = torch.from_numpy(test_img.astype(np.float32)).to(device)
                test_output = model(test_img).cpu().numpy().squeeze()
                test_output = np.argmax(np.transpose(test_output, (1, 2, 0)), axis=2)

                single_c_m = np.zeros(shape=(21, 21), dtype=np.uint32)
                IoU = np.zeros(shape=21, dtype=np.float32)
                single_pixel_accuracy, single_mIoU = 0., 0.
                count = 0
                for p in range(256):
                    for q in range(256):
                        pred = test_output[p, q]
                        gt = test_gts[j, p, q]
                        single_c_m[pred,gt] += 1
                        total_c_m[pred, gt] += 1

                for k in range(21):
                    union = np.sum(single_c_m[:, k]) + np.sum(single_c_m[k, :]) - single_c_m[k, k]
                    single_pixel_accuracy += single_c_m[k, k]
                    if np.sum(single_c_m[:, k]) != 0:
                        IoU[k] = single_c_m[k, k] / union
                        count += 1
                single_pixel_accuracy /= (256*256)
                single_mIoU = np.sum(IoU) / count
                learning_rate = optimizer.param_groups[0]['lr']
                os.makedirs(f'/home/aivs/바탕화면/hdd/kse/output/{try_name}_{i}', exist_ok=True)
                cv2.imwrite(f'/home/aivs/바탕화면/hdd/kse/output/{try_name}_{i}/{test_names[j]}({single_pixel_accuracy:.2f}, {single_mIoU:.4f}).jpg', index_to_color(test_output))


        for k in range(21):
            union = sum(total_c_m[:, k]) + sum(total_c_m[k, :]) - total_c_m[k, k]
            total_mIoU += (total_c_m[k, k] / union)

        total_pixel_accuracy = np.sum(total_c_m.diagonal()) / (test_n * 256 * 256)
        total_mIoU /= 21

        test_time = time.time()
        elapsed_time = (test_time - start_time) / 3600
        learning_rate = optimizer.param_groups[0]['lr']
        f2.write(f'{i}\t{total_pixel_accuracy}\t{total_mIoU}\n')
        print(f'step: {i}\t||  pixel accuracy: {total_pixel_accuracy:.4f},\tmIoU: {total_mIoU:.4f},\tlr: {learning_rate:.6f},\ttime: {elapsed_time:.1f}')


    f1.close()
    f2.close()
