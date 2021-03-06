# Python实现逻辑斯蒂回归

[逻辑回归与加正则化的逻辑斯蒂回归介绍](http://www.ai-start.com/ml2014/html/week3.html)

本实验室根据两次考试成绩与是否通过的数据，通过logistic回归，最后获得一个分类器。

## 逻辑斯蒂回归

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

![](https://raw.githubusercontent.com/xiaohuzai/ML-Excercise/master/pictures/chapter2/%E5%AE%9E%E9%AA%8C%E5%9B%BE/%E9%80%BB%E8%BE%91%E6%96%AF%E8%92%82%E6%95%B0%E6%8D%AE1.png)

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

逻辑斯蒂回归函数设为为：

![](https://raw.githubusercontent.com/xiaohuzai/ML-Excercise/master/pictures/chapter2/%E5%AE%9E%E9%AA%8C%E5%9B%BE/%E9%80%BB%E8%BE%91%E6%96%AF%E8%92%82%E5%9B%9E%E5%BD%92%E5%81%87%E8%AE%BE.PNG)

```python
# 定义sigmoid函数
def sigmoid(z):
    return(1 / (1 + np.exp(-z)))
```

损失函数为：

![](https://raw.githubusercontent.com/xiaohuzai/ML-Excercise/master/pictures/chapter2/%E5%AE%9E%E9%AA%8C%E5%9B%BE/%E9%80%BB%E8%BE%91%E6%96%AF%E8%92%82%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.PNG)

向量化的损失函数（矩阵形式）：

![](https://raw.githubusercontent.com/xiaohuzai/ML-Excercise/master/pictures/chapter2/%E5%AE%9E%E9%AA%8C%E5%9B%BE/%E9%80%BB%E8%BE%91%E6%96%AF%E8%92%82%E5%90%91%E9%87%8F%E5%8C%96%E7%9A%84%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.PNG)

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

![](https://raw.githubusercontent.com/xiaohuzai/ML-Excercise/master/pictures/chapter2/%E5%AE%9E%E9%AA%8C%E5%9B%BE/%E9%80%BB%E8%BE%91%E6%96%AF%E8%92%82%E5%81%8F%E5%AF%BC.PNG)

向量化的偏导（梯度）

![](https://raw.githubusercontent.com/xiaohuzai/ML-Excercise/master/pictures/chapter2/%E5%AE%9E%E9%AA%8C%E5%9B%BE/%E9%80%BB%E8%BE%91%E6%96%AF%E8%92%82%E5%90%91%E9%87%8F%E5%8C%96%E7%9A%84%E5%81%8F%E5%AF%BC.PNG)

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

[minimize()函数的介绍](https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.minimize.html)

[Jacobian矩阵和Hessian矩阵介绍](http://jacoxu.com/jacobian%E7%9F%A9%E9%98%B5%E5%92%8Chessian%E7%9F%A9%E9%98%B5/)

```python
# 计算损失函数得最小值。
from scipy.optimize import minimize
# 规定最大迭代次数为400次
res = minimize(costFunction, initial_theta, args=(X,y), jac=gradient, options={'maxiter':400})
res
# minimize()返回的格式是固定的，fun为costFunction函数迭代求得的最小值，hess_inv和jac分别为求得最小值时海森矩阵和雅克比矩阵的值，小写字母x为当costFunction函数最小时函数的解，在costFunction中即为theta的解。即计算损失函数最小时theta的值为[-25.16133284,   0.2062317 ,   0.2014716 ]。
Out[17]: 
      fun: 0.20349770158944375
 hess_inv: array([[  3.31474479e+03,  -2.63892205e+01,  -2.70237122e+01],
       [ -2.63892205e+01,   2.23869433e-01,   2.02682332e-01],
       [ -2.70237122e+01,   2.02682332e-01,   2.35335117e-01]])
      jac: array([ -9.52476821e-09,  -9.31921318e-07,  -2.82608930e-07])
  message: 'Optimization terminated successfully.'
     nfev: 31
      nit: 23
     njev: 31
   status: 0
  success: True
        x: array([-25.16133284,   0.2062317 ,   0.2014716 ])
```

即损失函数最小时theta的值：

```python
theta = res.x.T
theta
Out[20]: array([-25.16133284,   0.2062317 ,   0.2014716 ])
```

**咱们来看看考试1得分45，考试2得分85的同学通过概率有多高**

```python
sigmoid(np.array([1,45,85]).dot(theta))
Out[22]: 0.77629072405889421
```

即考试1得分45，考试2得分85的同学通过概率约为0.7763。

**画出决策边界**

```python
# 标注考试1得分45，考试2得分85的同学
plt.scatter(45, 85, s=60, c='r', marker='v', label='(45, 85)')
# plotData为之前定义的画分类点的函数
plotData(data, 'Exam 1 score', 'Exam 2 score', 'Admitted', 'Not admitted')
# 生成网格数据
x1_min, x1_max = X[:,1].min(), X[:,1].max(),
x2_min, x2_max = X[:,2].min(), X[:,2].max(),
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
# 计算h的值
h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0],1)), xx1.ravel(), xx2.ravel()].dot(res.x))
h = h.reshape(xx1.shape)
# 作等高线，可理解为在这条线上h值为0.5
plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
```

![](https://raw.githubusercontent.com/xiaohuzai/ML-Excercise/master/pictures/chapter2/%E5%AE%9E%E9%AA%8C%E5%9B%BE/%E9%80%BB%E8%BE%91%E6%96%AF%E8%92%82%E5%9B%9E%E5%BD%92%E8%BE%B9%E7%95%8C.png)

## 加正则化项的逻辑斯蒂回归

**导入数据**

``` python
data2 = loaddata('data2.txt', ',')
```

data2的格式为：

Dimensions:  (118, 3)
[[-0.092742  0.68494   1.      ]
 [-0.21371   0.69225   1.      ]
 [-0.375     0.50219   1.      ]
 [-0.51325   0.46564   1.      ]
 [-0.52477   0.2098    1.      ]]

**作分布图**

```python
y = np.c_[data2[:,2]]
X = data2[:,0:2]
plotData(data2, 'Microchip Test 1', 'Microchip Test 2', 'y = 1', 'y = 0')
```

![](https://raw.githubusercontent.com/xiaohuzai/ML-Excercise/master/pictures/chapter2/%E5%AE%9E%E9%AA%8C%E5%9B%BE/%E5%8A%A0%E6%AD%A3%E5%88%99%E5%8C%96%E9%A1%B9%E7%9A%84%E9%80%BB%E8%BE%91%E6%96%AF%E8%92%82%E5%9B%9E%E5%BD%92%E6%95%B0%E6%8D%AE%E5%88%86%E5%B8%831.png)

在上一个逻辑回归试验中，我们把sigmoid函数（即这里的g函数）设置的为简单的一次多项式。

![](https://raw.githubusercontent.com/xiaohuzai/ML-Excercise/master/pictures/chapter2/%E5%AE%9E%E9%AA%8C%E5%9B%BE/%E9%80%BB%E8%BE%91%E6%96%AF%E8%92%82%E5%9B%9E%E5%BD%92%E5%81%87%E8%AE%BE.PNG)

在这个逻辑回归实验里，因为样本的分布比较复杂，可以采用多次多项式来代替ΘTX。这里取最高六次项。

**取高阶多项式放入sigmoid函数进行模拟**

```python
from sklearn.preprocessing import PolynomialFeatures
# 生成一个六次多项式
poly = PolynomialFeatures(6)
# XX为生成的六次项的数据
XX = poly.fit_transform(data2[:,0:2])

# 六次项后有28个特征值了。即，之前我们只有两个特征值x1、x2，取六次项多项式后我们会有x1、x2、x1^2、x2^2、x1*x2、x1^2*x2、……，总共28项。
XX.shape
Out[12]: (118, 28)
```

**正则化**

因为取得的多项式最高项为6次，容易发生过拟合情况。将损失函数采取“**正则化**”处理，引入惩罚项。

正则化后损失函数：

![](https://raw.githubusercontent.com/xiaohuzai/ML-Excercise/master/pictures/chapter2/%E5%AE%9E%E9%AA%8C%E5%9B%BE/%E9%80%BB%E8%BE%91%E6%96%AF%E8%92%82%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.PNG)

向量化的损失函数（矩阵形式）：

![](https://raw.githubusercontent.com/xiaohuzai/ML-Excercise/master/pictures/chapter2/%E5%AE%9E%E9%AA%8C%E5%9B%BE/%E9%80%BB%E8%BE%91%E6%96%AF%E8%92%82%E5%90%91%E9%87%8F%E5%8C%96%E7%9A%84%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.PNG)

```python
# 定义损失函数
def costFunctionReg(theta,reg ,XX, y):
    m = y.size
    h = sigmoid(XX.dot(theta))
    J = -1.0*(1.0/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y)) +(reg/(2.0*m))*np.sum(np.square(theta[1:]))
    if np.isnan(J[0]):
        return(np.inf)
    return(J[0])
```



与之对应的偏导（梯度）：

![](https://raw.githubusercontent.com/xiaohuzai/ML-Excercise/master/pictures/chapter2/%E5%AE%9E%E9%AA%8C%E5%9B%BE/%E6%AD%A3%E5%88%99%E5%8C%96%E6%A2%AF%E5%BA%A6.png)

向量化的偏导（梯度）：

![](https://raw.githubusercontent.com/xiaohuzai/ML-Excercise/master/pictures/chapter2/%E5%AE%9E%E9%AA%8C%E5%9B%BE/%E6%AD%A3%E5%88%99%E5%8C%96%E6%A2%AF%E5%BA%A6%E7%9A%84%E5%90%91%E9%87%8F%E5%8C%96%E8%A1%A8%E8%BE%BE.png)

*注意，我们另外自己加的参数 θ0 不需要被正则化*

```python
# 定义正则化损失函数的偏导
def gradientReg(theta, reg, XX, y):
    m = y.size
    h = sigmoid(XX.dot(theta.reshape(-1,1)))
    grad = (1.0/m)*XX.T.dot(h-y) + (reg/m)*np.r_[[[0]],theta[1:].reshape(-1,1)]
    return(grad.flatten())
```

设初始化theta为0向量，计算此时初始损失值

```python
initial_theta = np.zeros(XX.shape[1])
costFunctionReg(initial_theta, 1, XX, y)

Out[9]: 0.69314718055994529
```

**画出决策边界**

定义预测函数，用来统计准确率。分类的阈值定为0.5，即计算的h(x)>0.5则分到1类（即通过），h(x)<0.5则分到0类（即不通过）：

```python
def predict(theta, X, threshold=0.5):
    h = sigmoid(X.dot(theta.T)) >= threshold
    # 返回的h值只会有两种，1或0
    return(h.astype('int'))
```

决策边界，咱们分别来看看正则化系数lambda太大太小分别会出现什么情况

> - Lambda = 0 : 就是没有正则化，这样的话，就过拟合咯
> - Lambda = 1 : 这才是正确的打开方式
> - Lambda = 100 : 卧槽，正则化项太激进，导致基本就没拟合出决策边界

```python
fig, axes = plt.subplots(1,3, sharey = True, figsize=(17,5))

# 分别取lambda为0、1、100
for i, C in enumerate([0.0, 1.0, 100.0]):
    # 最优化 costFunctionReg
    res2 = minimize(costFunctionReg, initial_theta, args=(C, XX, y), jac=gradientReg, options={'maxiter':3000})
    
    # 准确率
    accuracy = 100.0*sum(predict(res2.x, XX) == y.ravel())/y.size    

    # 对X,y的散列绘图
    plotData(data2, 'Microchip Test 1', 'Microchip Test 2', 'y = 1', 'y = 0', axes.flatten()[i])
    
    # 画出决策边界
    x1_min, x1_max = X[:,0].min(), X[:,0].max(),
    x2_min, x2_max = X[:,1].min(), X[:,1].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    h = sigmoid(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(res2.x))
    h = h.reshape(xx1.shape)
    axes.flatten()[i].contour(xx1, xx2, h, [0.5], linewidths=1, colors='g');       
    axes.flatten()[i].set_title('Train accuracy {}% with Lambda = {}'.format(np.round(accuracy, decimals=2), C))
```

![](https://raw.githubusercontent.com/xiaohuzai/ML-Excercise/master/pictures/chapter2/%E5%AE%9E%E9%AA%8C%E5%9B%BE/%E9%80%BB%E8%BE%91%E6%96%AF%E8%92%82%E6%AD%A3%E5%88%99%E5%8C%96%E5%88%86%E7%95%8C%E7%BB%93%E6%9E%9C.png)

