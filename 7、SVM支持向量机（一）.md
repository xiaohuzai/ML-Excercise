# SVM支持向量机（一）

在这里介绍一下监督学习中经典的分类模型SVM支持向量机。

### 最大间隔分类

我们依旧用这个最常见的iris花瓣数据集来做个试验。iris共有四个属性，取petal length, petal width两个作为分类的属性；iris共有三个类别，取setosa（y=0）和versicolor（y=1）两个类别进行二分类。使用SVC分类函数，选择线性分类器，惩罚参数设为正无穷，即所有点都能正确分类。

```python
from sklearn.svm import SVC
from sklearn import datasets

iris = datasets.load_iris()
#iris共有四个属性，取petal length, petal width两个作为分类的属性，目的是好作图
X = iris["data"][:, (2, 3)]
y = iris["target"]
#iris共有三个类别，取setosa（y=0）和versicolor（y=1）两个类别进行二分类。
setosa_or_versicolor = (y == 0) | (y == 1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]

# SVM Classifier model，核函数选择线性，惩罚参数为正无穷，即选择让所有样本点都满足条件
svm_clf = SVC(kernel="linear", C=float('inf'))
svm_clf.fit(X, y)

Out[3]: 
SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
```

做图：

```python
import numpy as np
import matplotlib.pyplot as plt

# 函数作用：绘制分界超平面和边界距离
def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    # w，b为训练得到的SVM超平面系数
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # 决策边界 w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    # SVM分类超平面
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]

    # 边界距离
    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    # svs为训练得到的支持向量点
    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)

plt.figure(figsize=(12,2.7))
plt.subplot(122)
plot_svc_decision_boundary(svm_clf, 0, 5.5)
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris-Versicolor")
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris-Setosa")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="upper left", fontsize=14)
plt.axis([0, 5.5, 0, 2])

#save_fig("large_margin_classification_plot")
plt.show()
```

![](C:\ML-Excercise\pictures\SVM\1.png)

上图中标志的点即为支持向量点。

### 最大间隔与容错

刚刚我们使用的数据很清楚的能够被线性可分，但现实中往往数据不是那么容易用一个线性分类器就可以分开的，这时候需要引入容错。

还是以鸢尾花数据集为例。代码里面使用Pipline设置了一个工作流，先对数据进行标准化`StandardScaler`，再进行线性SVM分类`LinearSVC`，其中惩罚参数C设为1.

```python
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
# 选取petal length, petal width两个属性
X = iris["data"][:, (2, 3)]  
# 原数据分成了三类，分别是setosa（y=0）、versicolor（y=1）、Vignica(y=2)
# 现在将数据集处理，分成两类，分别是非Virginica(y=0)和Viginica(y=1)
y = (iris["target"] == 2).astype(np.float64)  

# 关于Pipline的简介https://blog.csdn.net/lijingpengchina/article/details/52296650
# 简单来说，Pipeline可以将许多算法模型串联起来，比如将特征提取、归一化、分类组织在一起形成一个典型的机器学习问题工作流
svm_clf = Pipeline([
        ("scaler", StandardScaler()), #先标准化
        ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42)), #后进行SVC线性分类，设置损失参数C=1
    ])

# 训练模型
svm_clf.fit(X, y)

Out[14]: 
Pipeline(memory=None,
     steps=[('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('linear_svc', LinearSVC(C=1, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='hinge', max_iter=1000, multi_class='ovr',
     penalty='l2', random_state=42, tol=0.0001, verbose=0))])
```

使用训练的模型对petal length=5.5, petal width=1.7进行预测，结果显示类别为1，即为Viginica

```python
svm_clf.predict([[5.5, 1.7]])
Out[15]: array([1.])
```

我们再来试一下不同的正则化强度。惩罚参数C分别设为1和100.

```python
scaler = StandardScaler()
svm_clf1 = LinearSVC(C=1, loss="hinge", random_state=42)
svm_clf2 = LinearSVC(C=100, loss="hinge", random_state=42)

scaled_svm_clf1 = Pipeline([
        ("scaler", scaler),
        ("linear_svc", svm_clf1),
    ])
scaled_svm_clf2 = Pipeline([
        ("scaler", scaler),
        ("linear_svc", svm_clf2),
    ])

scaled_svm_clf1.fit(X, y)
scaled_svm_clf2.fit(X, y)
```

因为我们在训练的时候点经过了标准化，所以在画图的时候要把标准化变换回原来样本的状态，可以理解为"反标准化"。同时，在上一节绘制超平面和边界距离的函数`plot_svc_decision_boundary`中，我们使用了`svm_clf.support_vectors_`来获取支持向量点，`svm_clf`是由`SVC`函数训练出来的，然而在我们刚刚使用的是`LinearSVC`函数训练模型，而`LinearSVC`对象是没有`.support_vectors_`属性（即支持向量点），所以我们需要自己来定义。

```python
# Convert to unscaled parameters
b1 = svm_clf1.decision_function([-scaler.mean_ / scaler.scale_])
b2 = svm_clf2.decision_function([-scaler.mean_ / scaler.scale_])
w1 = svm_clf1.coef_[0] / scaler.scale_
w2 = svm_clf2.coef_[0] / scaler.scale_
svm_clf1.intercept_ = np.array([b1])
svm_clf2.intercept_ = np.array([b2])
svm_clf1.coef_ = np.array([w1])
svm_clf2.coef_ = np.array([w2])

# Find support vectors (LinearSVC does not do this automatically)

# y的取值是[0,1]，t的取值变为[-1,1]，符合SVM的一般形式
t = y * 2 - 1
# 我们在这里设定在边界内的点都是support_vector
support_vectors_idx1 = (t * (X.dot(w1) + b1) < 1).ravel()
support_vectors_idx2 = (t * (X.dot(w2) + b2) < 1).ravel()
svm_clf1.support_vectors_ = X[support_vectors_idx1]
svm_clf2.support_vectors_ = X[support_vectors_idx2]
```

分别作出C=1和C=100的SVM线性分类图。

```python
plt.figure(figsize=(12,3.2))
plt.subplot(121)
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^", label="Iris-Virginica")
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs", label="Iris-Versicolor")
plot_svc_decision_boundary(svm_clf1, 1, 7)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="upper left", fontsize=14)
plt.title("$C = {}$".format(svm_clf1.C), fontsize=16)

plt.subplot(122)
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
plot_svc_decision_boundary(svm_clf2, 1, 7)
plt.xlabel("Petal length", fontsize=14)
plt.title("$C = {}$".format(svm_clf2.C), fontsize=16)

```

![](C:\ML-Excercise\pictures\SVM\2.png)

使用`plt.axis([4, 6, 0.8, 2.8])`可以讲图幅放大到`[4,6,0.8,2.8]`区域，看的更清晰。可以看出C=100时，边界距离明显比C=1要小，更关心数据划分的正确性，划分错误的点要少一些。

![](C:\ML-Excercise\pictures\SVM\3.png)

