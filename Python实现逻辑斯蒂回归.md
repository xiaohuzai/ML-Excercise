# Python实现逻辑斯蒂回归

本实验室根据两次考试成绩与通过的数据，通过logistic回归，最后获得一个分类器。

**导入数据**

```python
import numpy as np
def loaddata(file, delimeter):
    #以delimeter为分隔符导入file数据
    data = np.loadtxt(file, delimiter=delimeter)
    print('Dimensions: ',data.shape)
    # 打印数据前6行
    print(data[1:6,:])
    return(data)
data = loaddata('data1.txt', ',')
```

结果为：

Dimensions:  (100, 3)
[[30.28671077 43.89499752  0.        ]
 [35.84740877 72.90219803  0.        ]
 [60.18259939 86.3085521   1.        ]
 [79.03273605 75.34437644  1.        ]
 [45.08327748 56.31637178  0.        ]]

可以看见data的数据结构为一个100*3的矩阵，第一列为exam1的成绩，第二列为exam2的成绩，第三列为是否最终通过（0为否，1为是）。

```python
# 作图显示数据分布
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

plotData(data, 'Exam 1 score', 'Exam 2 score', 'Pass', 'Fail')
```

![](D:\ML-Excercise\pictures\chapter2\实验图\逻辑斯蒂数据1.png)

读取数据作为X与y向量：

```python
X = np.c_[np.ones((data.shape[0],1)), data[:,0:2]]
y = np.c_[data[:,2]]
```

则X为：

1	34.6237	78.0247
1	30.2867	43.895
1	35.8474	72.9022
1	60.1826	86.3086
1	79.0327	75.3444
1	45.0833	56.3164
1	61.1067	96.5114
1	75.0247	46.554
1	76.0988	87.4206
1	84.4328	43.5334......

y为：

0
0
0
1
1
0
1
1
1
1......

**逻辑斯蒂回归**

逻辑斯蒂回归假设为：

![](D:\ML-Excercise\pictures\chapter2\实验图\逻辑斯蒂回归假设.PNG)

```python
# 定义sigmoid函数
 def sigmoid(z):
    return(1 / (1 + np.exp(-z)))
```

损失函数为：

![](D:\ML-Excercise\pictures\chapter2\实验图\逻辑斯蒂损失函数.PNG)

向量化的损失函数（矩阵形式）：

![](D:\ML-Excercise\pictures\chapter2\实验图\逻辑斯蒂向量化的损失函数.PNG)

```python
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
cost = costFunction(initial_theta, X, y)
print('Cost: \n', cost)
```

Cost: 
 0.6931471805599452

求偏导（梯度）

![](D:\ML-Excercise\pictures\chapter2\实验图\逻辑斯蒂偏导.PNG)

向量化的偏导（梯度）

![](D:\ML-Excercise\pictures\chapter2\实验图\逻辑斯蒂向量化的偏导.PNG)

```python
# 求梯度
def gradient(theta, X, y):
    m = y.size
    # 此处h为一100*1的列向量
    h = sigmoid(X.dot(theta.reshape(-1,1)))
    # grad为一个3*100的矩阵
    grad =(1.0/m)*X.T.dot(h-y)
    return(grad.flatten())
#此时逻辑斯蒂函数在初始化theta处在Θ0、Θ1、Θ2处的梯度分别为：
grad = gradient(initial_theta, X, y)
print('Grad: \n', grad)
```

Grad: 
 [ -0.1        -12.00921659 -11.26284221]

**最小化损失函数**

