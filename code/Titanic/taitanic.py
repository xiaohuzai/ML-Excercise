# -*- coding: utf-8 -*-
"""
Created on Fri May 25 23:12:20 2018

@author: huang
"""

import pandas as pd 
import numpy as np 
from pandas import Series,DataFrame

#读取数据
data_train = pd.read_csv("Train.csv")

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文
plt.rcParams['axes.unicode_minus']= False  #用来正确显示负号
                                                                                  

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"各乘客等级的获救情况")
plt.xlabel(u"乘客等级") 
plt.ylabel(u"人数") 

plt.show()







