import os
import numpy as np
# import matplotlib.pyplot as plt
import os, sys

# from glob import glob
# import pandas as pd
# from tqdm import tqdm

import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skimage import transform

dir = os.listdir("data_pub/train")
Sizes = []
data = []
data2 = []
labels2 = []
labels = []
a, b, c, d = 0, 0, 0,0


Numbers = []

k = 0
# for i in range(0, 100 ):
#     if i % 2 == 0:
#         TempArray = np.zeros((4, a, b, c))
#         # Temp = np.load("data_pub/train/" + dir[i])
#         # Temp = transform.resize(Temp, (4, 128, 128, 128))
#         # # # shape = np.shape(Temp)
#         # # # TempArray[:,:shape[1],:shape[2], :shape[3] ] = Temp
#         # # print(np.shape(Temp))
#         # # data.append(Temp)
#         # # Temp = torch.FloatTensor(Temp)
#         # np.save("tt/" + dir[i],Temp)
#     else:
#         # TempArray = np.zeros((a, b, c))
#         Temp = np.load("data_pub/train/" + dir[i])
#         s = np.shape(Temp)
#         for ii in range(s[0]):
#             for jj in range(s[1]):
#                 for kk in range(s[2]):
#                     c = int(Temp[ii][jj][kk])
#                     Arr[ii, jj, kk, c] += 1
#         print(Arr[100, 100, 100])

Arr = np.load("Arr.npy")
s = np.shape(Arr)
newArr = np.zeros((s[0],s[1],s[2]))
for ii in range(s[0]):
    for jj in range(s[1]):
        for kk in range(s[2]):
            C = Arr[ii][jj][kk]
            if C[1] > 6 or C[2]>6 or C[3]>6:

                index = np.argmax(C[1:]) + 1
                newArr[ii][jj][kk] = index
np.save("newArr.npy", newArr)
print("Done")
        # Vol = s[0] * s[1] * s[2]
        # a += Vol - np.count_nonzero(Temp)
        # b += sum(Temp[Temp == 1.0])
        # c += sum(Temp[Temp == 2.0])/2
        # d += sum(Temp[Temp == 3.0])/3
        # print(a, b, c, d)

        # Temp = transform.resize(Temp, ( 1, 128, 128, 128))
        # # Temp = torch.FloatTensor(Temp)
        # np.save("tt/" + dir[i], Temp)
        #
        # labels.append(Temp)
print(a/200000, b/200000, c/200000, d/200000)


#
# data = torch.FloatTensor(data)
# labels = torch.LongTensor(np.array(labels))
# labels = F.one_hot(labels).view(labels.size()[0], 4,labels.size()[1], labels.size()[2], labels.size()[3])
# print(labels.size())
# data2 = torch.FloatTensor(data2)
# mean, std = np.mean(data), np.std(data)
# print(mean, std)
# torch.save(data, "Saud_data.pt")
# torch.save(labels, "Saud_labels.pt")
# # torch.save(data2, "data2.pt")
print("DATA NUMBER "+ str(k) + ": DONE!")

# labels = torch.FloatTensor(labels)
# labels2 = torch.FloatTensor(labels2)
# mean, std = np.mean(data), np.std(data)
# print(mean, std)
# torch.save(labels2, "labels2.pt")
# print("n")

# labels = np.array(labels)
# mean2, std2 = np.mean(labels), np.std(labels)
# print(mean2, std2)
# np.save("labels_mini.npy", labels)
# print("n")
#
# np.save("Sizes_mini.npy", np.array(Sizes))
