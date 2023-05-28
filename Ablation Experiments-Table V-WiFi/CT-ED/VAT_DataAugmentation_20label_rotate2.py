import torch
import numpy as np
from sklearn.model_selection import train_test_split
import random

def TrainDataset(ft, random_num):
    x = np.load(f"/data/fuxue/WiFi/Dataset/Feet{ft}_X_train.npy")
    y = np.load(f"/data/fuxue/WiFi/Dataset/Feet{ft}_Y_train.npy")
    y = y.astype(np.uint8)
    X_train_labeled1, X_train_unlabeled1, Y_train_labeled1, Y_train_unlabeled1 = train_test_split(x, y, test_size=0.5, random_state=random_num)
    X_train_labeled2, X_train_unlabeled2, Y_train_labeled2, Y_train_unlabeled2 = train_test_split(X_train_labeled1,Y_train_labeled1, test_size=0.6, random_state=random_num)

    X_train_label, X_val, Y_train_label, Y_val = train_test_split(X_train_labeled2, Y_train_labeled2, test_size=0.3, random_state=random_num)

    X_train_unlabeled = np.concatenate((X_train_unlabeled1,X_train_unlabeled2), axis=0)
    Y_train_unlabeled = np.concatenate((Y_train_unlabeled1, Y_train_unlabeled2), axis=0)

    X_train = np.concatenate((X_train_label, X_train_unlabeled), axis=0)
    Y_train = np.concatenate((Y_train_label, Y_train_unlabeled), axis=0)

    return X_train_label, X_train_unlabeled, X_val, Y_train_label, Y_train_unlabeled, Y_val

def rotate_matrix(theta):
    m = np.zeros((2,2))
    m[0, 0] = np.cos(theta)
    m[0, 1] = -np.sin(theta)
    m[1, 0] = np.sin(theta)
    m[1, 1] = np.cos(theta)
    return m

def Rotate_DA(data, target):
    x_DA_all = np.zeros((1, data.shape[1], data.shape[2]))
    y_DA_all = np.zeros((1,))
    for i in range(data.shape[0]):
        x = data[i]
        y = target[i]
        x = x.transpose(1, 0)
        x = np.expand_dims(x, axis=0)
        x_rotate1 = np.matmul(x, rotate_matrix(np.pi/2))
        x_rotate2 = np.matmul(x, rotate_matrix(np.pi))
        x_rotate3 = np.matmul(x, rotate_matrix(3*np.pi/2))

        x = x.transpose(0, 2, 1)
        x_rotate1 = x_rotate1.transpose(0,2,1)
        x_rotate2 = x_rotate2.transpose(0,2,1)
        x_rotate3 = x_rotate3.transpose(0,2,1)
        x_DA = np.vstack((x, x_rotate1, x_rotate2, x_rotate3))

        y_DA = np.tile(y, (1, 4))
        y_DA = y_DA.T
        y_DA = y_DA.reshape(-1)
        y_DA = y_DA.T

        x_DA_all = np.concatenate([x_DA_all, x_DA], axis = 0)
        y_DA_all = np.concatenate([y_DA_all, y_DA], axis = 0)

    return x_DA_all[1:x_DA_all.shape[0]], y_DA_all[1:y_DA_all.shape[0]]

def rand_bbox(size,lamb):
    length = size[2]
    cut_rate = 1.-lamb
    cut_length = np.int(length*cut_rate)
    cx = np.random.randint(length)
    bbx1 = np.clip(cx - cut_length//2, 0, length)
    bbx2 = np.clip(cx + cut_length//2, 0, length)
    return bbx1, bbx2

def StrongAugmentation(data, target):
    lam = np.random.beta(1, 1)
    bbx1, bbx2 = rand_bbox(data.size(), lam)
    data, target = Rotate_DA(np.array(data.cpu()), np.array(target.cpu()))
    data = torch.tensor(data).float().cuda()
    target = torch.tensor(target).long().cuda()
    data[:, :, bbx1: bbx2] = torch.zeros((data.size()[1],bbx2-bbx1)).cuda()
    return  data, target

def WeakAugmentation(data, target):
    data, target = Rotate_DA(np.array(data.cpu()), np.array(target.cpu()))
    data = torch.tensor(data).float().cuda()
    target = torch.tensor(target).long().cuda()
    return  data, target

def TestDataset(ft):
    x = np.load(f"/data/fuxue/WiFi/Dataset/Feet{ft}_X_test.npy")
    y = np.load(f"/data/fuxue/WiFi/Dataset/Feet{ft}_Y_test.npy")
    y = y.astype(np.uint8)
    return x, y

