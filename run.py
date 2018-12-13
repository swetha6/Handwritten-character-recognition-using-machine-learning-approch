import numpy as np
from sklearn.model_selection import train_test_split
import mlp
import cv2
import torch
import torch.utils.data
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as fun

np.set_printoptions(threshold=np.nan)
# #load data
# data = np.load('ClassData.npy')
# labels = np.load('ClassLabels.npy')
# new_arr=[]
# for i in range(data.size):
#     img = data[i]
#     res = cv2.resize(img, dsize=(50, 50), interpolation=cv2.INTER_CUBIC)
#     new_arr.append(res)
# data=np.array(new_arr)
# final_data=[]
# final_label=[]
# for i in range(labels.size):
#     if(labels[i]==1 or labels[i]==2):
#         final_data.append(data[i])
#         final_label.append(labels[i])

# data = np.array(final_data)
# labels = np.array(final_label)

# data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
#  train_data = data[0::2, :]
#     validation_data = data[1::4, :]
#     x_test = data[3::4, :]

#     train_label = target[0::2, :]
#     validation_labels = target[1::4, :]
#     y_test = target[3::4, :]
data = np.load('finalAB.npy')
labels = np.load('final_labelAB.npy')

def mlp_process(data, labels):
    w = 2
    h = len(data)
    target = [[0 for model in range(w)] for y in range(h)]
    target = np.array(target)
    # print(labels)

    for i in range(data.shape[0]):
        if(labels[i] == 1):
            target[i,0] = 1
        elif(labels[i] == 2):
            target[i,1] = 1
        # elif(labels[i] == 3):
        #     target[i,2] = 1
        # elif(labels[i] == 4):
        #     target[i,3] = 1
        # elif(labels[i] == 5):
        #     target[i,4] = 1
        # elif(labels[i] == 6):
        #     target[i,5] = 1
        # elif(labels[i] == 7):
        #     target[i,6] = 1
        # elif(labels[i] == 8):
        #     target[i,7] = 1

    # print(target)
    #split the data into train, test and validation data for mlp
    #Set up Neural Network
    # x_train, y_train, x_valid, y_valid = train_test_split(data, target, test_size = 0.3, random_state = 0)
    x_train = data[0::2, :]
    x_valid = data[1::4, :]
    x_test = data[3::4, :]

    y_train = target[0::2, :]
    y_valid = target[1::4, :]
    y_test = target[3::4, :]

    hidden_layers = 50
    learning_rate = 0.1
    net = mlp.mlp(x_train,y_train,hidden_layers,outtype='softmax')
    net.earlystopping(x_train,y_train,x_valid,y_valid,learning_rate)
    net.confmat(x_test,y_test)
    return net

net = mlp_process(data, labels)
