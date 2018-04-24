# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 23:27:20 2018

@author: huang
"""
import numpy as np
def loaddata(file, delimeter):
    #以delimeter为分隔符导入file数据
    data = np.loadtxt(file, delimiter=delimeter)
    print('Dimensions: ',data.shape)
    # 打印数据前6行
    print(data[1:6,:])
    return(data)
data = loaddata('data1.txt', ',')

import matplotlib.pyplot as plt
def plotData(data, label_x, label_y, label_pos, label_neg, axes=None):
    # 获得正负样本的下标(即哪些是正样本，哪些是负样本)
    neg = data[:,2] == 0
    pos = data[:,2] == 1
    if axes == None:
        axes = plt.gca()
    axes.scatter(data[pos][:,0], data[pos][:,1], marker='+', c='k', s=60, linewidth=2, label=label_pos)
    axes.scatter(data[neg][:,0], data[neg][:,1], c='y', s=60, label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon= True, fancybox = True)
# plotData(data, 'Exam 1 score', 'Exam 2 score', 'Pass', 'Fail')

X = np.c_[np.ones((data.shape[0],1)), data[:,0:2]]
y = np.c_[data[:,2]]

# 定义sigmoid函数
def sigmoid(z):
    return(1 / (1 + np.exp(-z)))

# 定义损失函数
def costFunction(theta, X, y):
    m = y.size
    # 此处h为一个列向量
    h = sigmoid(X.dot(theta)) 
    J = -1.0*(1.0/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y))            
    return J[0]

# 初始化theta为0向量
initial_theta = np.zeros(X.shape[1])
# 此时逻辑斯蒂回归的损失函数值为
# cost = costFunction(initial_theta, X, y)
# print('Cost: \n', cost)

# 求梯度
def gradient(theta, X, y):
    m = y.size
    # 此处h为一100*1的列向量
    h = sigmoid(X.dot(theta.reshape(-1,1)))
    # grad为一个3*100的矩阵
    grad =(1.0/m)*X.T.dot(h-y)
    return(grad.flatten())
#此时逻辑斯蒂函数在初始化theta处在Θ0、Θ1、Θ2处的梯度分别为：
# grad = gradient(initial_theta, X, y)
# print('Grad: \n', grad)

# 计算损失函数得最小值。
from scipy.optimize import minimize
# 规定最大迭代次数为400次
res = minimize(costFunction, initial_theta, args=(X,y), jac=gradient, options={'maxiter':400})

theta = res.x.T

plt.scatter(45, 85, s=60, c='r', marker='v', label='(45, 85)')
plotData(data, 'Exam 1 score', 'Exam 2 score', 'Admitted', 'Not admitted')
x1_min, x1_max = X[:,1].min(), X[:,1].max(),
x2_min, x2_max = X[:,2].min(), X[:,2].max(),
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0],1)), xx1.ravel(), xx2.ravel()].dot(theta.T))
h = h.reshape(xx1.shape)
plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')




