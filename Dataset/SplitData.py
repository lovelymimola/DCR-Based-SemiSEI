import math
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import random
import json
import numpy as np
import scipy.io as scio

def Power_Normalization(x):
    for i in range(x.shape[0]):
        max_power = (np.power(x[i,0,:],2) + np.power(x[i,1,:],2)).max()
        x[i] = x[i] / np.power(max_power, 1/2)
    return x

def WiFi_Dataset_slice(ft):
    devicename = ['3123D7B', '3123D7D', '3123D7E', '3123D52', '3123D54', '3123D58', '3123D64', '3123D65',
                  '3123D70', '3123D76', '3123D78', '3123D79', '3123D80', '3123D89', '3123EFE', '3124E4A']
    data_IQ_wifi_all = np.zeros((1,2,6000))
    data_target_all = np.zeros((1,))
    target = 0
    for classes in range(16):
        for recoder in range(1):
            inputFilename = f'/data/fuxue/WiFi/Dataset/{ft}ft/WiFi_air_X310_{devicename[classes]}_{ft}ft_run{recoder+1}'
            with open("{}.sigmf-meta".format(inputFilename),'rb') as read_file:
                meta_dict = json.load(read_file)
            with open("{}.sigmf-data".format(inputFilename),'rb') as read_file:
                binary_data = read_file.read()
            fullVect = np.frombuffer(binary_data, dtype=np.complex128)
            even = np.real(fullVect) #提取复数信号中的实部
            odd = np.imag(fullVect)  #提取复数信号中的虚部
            length = 6000
            num = 0
            data_IQ_wifi = np.zeros((math.floor(len(even)/length), 2, 6000))
            data_target = np.zeros((math.floor(len(even)/length),))
            for begin in range(0,len(even)-(len(even)-math.floor(len(even)/length)*length),length):
                data_IQ_wifi[num,0,:] = even[begin:begin+length]
                data_IQ_wifi[num,1,:] = odd[begin:begin+length]
                data_target[num,] = target
                num = num + 1
            data_IQ_wifi_all = np.concatenate((data_IQ_wifi_all,data_IQ_wifi),axis=0)
            data_target_all = np.concatenate((data_target_all, data_target), axis=0)
        target = target + 1

    return data_IQ_wifi_all[1:,], data_target_all[1:,]

def SplitData(ft, random_num):
    x,y = WiFi_Dataset_slice(ft)
    x = Power_Normalization(x)
    print(x.shape)
    y = y.astype(np.uint8)
    x_, X_test, y_, Y_test = train_test_split(x, y, test_size=0.3, random_state=random_num)
    X_train, x__, Y_train, y__ = train_test_split(x_, y_, train_size=0.0825, random_state=random_num)
    print(X_test.shape)
    print(X_train.shape)
    np.save(f'Feet{ft}_X_train.npy', X_train)
    np.save(f'Feet{ft}_Y_train.npy', Y_train)

    np.save(f'Feet{ft}_X_test.npy', X_test)
    np.save(f'Feet{ft}_Y_test.npy', Y_test)

if __name__ == "__main__":
    SplitData(ft=62, random_num=30)