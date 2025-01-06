import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PSP_network import PSPNet
from PSP_function import data_load, mini_batch, index2color, color2index
import time
from tqdm import tqdm
import os
import cv2
from torch.profiler import profile, ProfilerActivity

VOC_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
               "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
               "potted plant", "sheep", "sofa", "train", "tv/monitor"]

path = '/home/aivs/바탕화면/hdd/dataset/VOC2012'
model_path = '/home/aivs/바탕화면/hdd/kse/pythonProject/model/resnet152_AL_aug80_5000.pt'
try_name = ('resnet152_noAL_aug80')

batch_size = 32
base_lr = 0.01
end_lr = 0.000001
power = 0.9
iter = 30000
start = 1
a = 0.4

# data load

# print('Datasets loading...')
# train, test, train_gts, test_gts = data_load(path)

# np.save('/home/aivs/바탕화면/hdd/kse/pythonProject/psp/train.npy', train)
# np.save('/home/aivs/바탕화면/hdd/kse/pythonProject/psp/test.npy', test)
# np.save('/home/aivs/바탕화면/hdd/kse/pythonProject/psp/train_gts.npy', train_gts)
# np.save('/home/aivs/바탕화면/hdd/kse/pythonProject/psp/test_gts.npy', test_gts)
#
train = np.load('train.npy')
test = np.load('test.npy')
train_gts = np.load('train_gts.npy')
test_gts = np.load('test_gts.npy')

test_names = open(path+'/ImageSets/Segmentation/val.txt', 'r').readlines()

for i in range(len(test_names)):
    test_names[i] = test_names[i].strip('\n')
train_n, test_n, class_num = len(train), len(test), len(VOC_CLASSES)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PSPNet().to(device)
# model.load_state_dict(torch.load(model_path))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
# optimizer = optim.Adam(model.parameters(), lr=base_lr)

start_t = time.time()

print('train start')

# train
for i in range(start, iter + 1):

    f1 = open(f'loss.txt', 'a+')
    f2 = open(f'{try_name}.txt', 'a+')
    model.train()

    optimizer.zero_grad()
    batch_img, batch_gts = mini_batch(train_n, batch_size, train, train_gts)
    batch_img, batch_gts = torch.from_numpy(batch_img).to(device), torch.from_numpy(batch_gts).to(device)
    # print(f"After block1: {torch.cuda.memory_allocated() / 1e6} MB")
    out= model(batch_img, False)
    # out, aux_out = model(batch_img, True)
    loss = criterion(out, batch_gts)
    # loss = criterion(out, batch_gts) + a * criterion(aux_out, batch_gts)
    loss.backward()
    optimizer.step()

    optimizer.param_groups[0]['lr'] = (base_lr - end_lr) * ((1.0 - i / iter) ** power) + end_lr

    if i % 500 == 0:
        print(f'{i}\t{loss.item():.6f}\t{optimizer.param_groups[0]["lr"]:.7f}')
        f1.write(f'{i}\t{loss.item():.6f}\n')

    # test
    if i % 5000 == 0:
        # torch.save(model.state_dict(), f'/home/aivs/바탕화면/hdd/kse/pythonProject/model/{try_name}_{i}.pt')
        model.eval()

        #initialize
        total_mIoU, total_pixel_accuracy = 0., 0.
        total_c_m = np.zeros(shape=(21, 21), dtype=np.uint32)

        with torch.no_grad():
            for j in tqdm(range(test_n), desc='testing...'):
                test_img = test[j:j+1, :, :, :] / 255.0 * 2.0 - 1.0
                test_img = np.transpose(test_img, (0, 3, 1, 2))
                test_img = torch.from_numpy(test_img.astype(np.float32)).to(device)
                test_output = model(test_img, False)
                test_output = test_output.cpu().numpy().squeeze()
                test_output = np.argmax(np.transpose(test_output, (1, 2, 0)), axis=2)

                test_gt = color2index(cv2.cvtColor(test_gts[j, :, :, :] , cv2.COLOR_BGR2RGB))
                single_c_m = np.zeros(shape=(21, 21), dtype=np.uint32)
                IoU = np.zeros(shape=21, dtype=np.float16)
                single_PA, single_mIoU = 0., 0.
                for p in range(256):
                    for q in range(256):
                        pred = test_output[p, q]
                        gt = test_gt[p, q]
                        single_c_m[pred,gt] += 1
                        total_c_m[pred, gt] += 1

                count = 0
                for k in range(21):
                    union = np.sum(single_c_m[:, k]) + np.sum(single_c_m[k, :]) - single_c_m[k, k]
                    single_PA += single_c_m[k, k]
                    if np.sum(single_c_m[:, k]) != 0:
                        IoU[k] = single_c_m[k, k] / union
                        count += 1
                single_PA /= ( 256 * 256 )
                single_mIoU = np.sum(IoU) / count

                if i == 30000:
                    os.makedirs(f'/home/aivs/바탕화면/hdd/kse/output/{try_name}', exist_ok=True)
                    cv2.imwrite(f'/home/aivs/바탕화면/hdd/kse/output/{try_name}/{test_names[j]}({single_PA:.3f}, {single_mIoU:.3f}).jpg', index2color(test_output))

        for k in range(21):
            union = sum(total_c_m[:, k]) + sum(total_c_m[k, :]) - total_c_m[k, k]
            IoU = total_c_m[k, k] / union
            total_mIoU += IoU
            f2.write(f'{IoU:.4f}\t')
            if k == 9: f2.write('\n')
        f2.write(f'\n')
        total_PA = np.sum(total_c_m.diagonal()) / (test_n * 256 * 256)
        total_mIoU /= 21

        test_t = time.time()
        elapsed_h = (test_t - start_t) // 3600
        elapsed_m = ((test_t - start_t) // 60) % 60
        lr = optimizer.param_groups[0]['lr']
        f2.write(f'{i}\t{total_PA:.4f}\t{total_mIoU:.4f}\n')

        print(f'step: %d\t|| PA: %.3f, mIoU: %.3f, lr: %f, time: %dh%dm\n' % (i, total_PA, total_mIoU, lr, elapsed_h, elapsed_m))

    f1.close()
    f2.close()
