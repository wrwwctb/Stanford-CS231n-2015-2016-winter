import numpy as np
import keras
from keras import backend
import matplotlib.pyplot as plt
#import platform, time, os

def calculate_performance(pre, tru, wt=1.): 
    '''

    pre, tru: 1D boolean ndarray. False is HS. True is PU

    pre: predicted
    tru: truth labels
    wt: float. weights

    Returns: (rHS, pHS, rPU, pPU, ePU)

    r is recall
    p is precision
    e is efficiency. eHS is the same as rHS

    (rHS, ePU) is the physics metric. (rHS, pHS) is the cs metric.

    rHS = (pre==0 & tru==0) / tru==0
    pHS = (pre==0 & tru==0) / pre==0
    rPU = (pre==1 & tru==1) / tru==1
    pPU = (pre==1 & tru==1) / pre==1
    ePU = (pre==0 & tru==1) / tru==1
    '''
    pre = np.array(pre) > 0 # numercial values would give wrong results
    tru = np.array(tru) > 0
    Npre = ~pre
    Ntru = ~tru
    temp1 = sum((Npre & Ntru) * wt)
    temp2 = sum(( pre &  tru) * wt)
    rHS = temp1 / sum(Ntru * wt)
    pHS = temp1 / sum(Npre * wt)
    sumtru = sum(tru * wt)
    rPU = temp2 / sumtru
    pPU = temp2 / sum( pre * wt)
    ePU = sum((Npre & tru) * wt) / sumtru
    return (rHS, pHS, rPU, pPU, ePU)

#aaa = [False, False, True, True]
#bbb = [False, True,  False, True]
#print(calculate_performance(aaa, bbb, [.4, .3, .2, .1]))

def rotate90(data):
    '''expect N C H W'''
    data = np.copy(data)
    data = np.swapaxes(data, 2, 3)
    data = data[:, :, ::-1, :]
    return data

def rotate90_4way_append(data0):
    '''expect N C H W'''
    data1 = rotate90(data0)
    data2 = rotate90(data1)
    data3 = rotate90(data2)
    data = np.concatenate((data0, data1, data2, data3))
    return data

def scale(image):
    temp = np.sum(image, axis=(2, 3), keepdims=True)
    #temp[temp == 0] = 1  # some frames are all zero
    temp = np.mean(temp, axis=0, keepdims=True)
    return temp

