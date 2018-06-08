# 8、SVM支持向量机（二）

虽然SVM在分类问题非常有效率，分类结果也很好，但是很多数据并不能用一个超平面就可以把他们分割开来。

```python
# 创造一个（-4,4）中间有九个点的等差数列，reshape(-1,1)意为将其转换为一列，不管它有多少行
X1D = np.linspace(-4, 4, 9).reshape(-1, 1)
# 垂直堆叠
X2D = np.c_[X1D, X1D**2]
# 类别
y = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])

plt.figure(figsize=(11, 4))

plt.subplot(121)
plt.grid(True, which='both')
# 画水平线
plt.axhline(y=0, color='k')
plt.plot(X1D[:, 0][y==0], np.zeros(4), "bs")
plt.plot(X1D[:, 0][y==1], np.zeros(5), "g^")
# 设置y轴不标注
plt.gca().get_yaxis().set_ticks([])
plt.xlabel(r"$x_1$", fontsize=20)
plt.axis([-4.5, 4.5, -0.2, 0.2])

plt.subplot(122)
plt.grid(True, which='both')
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.plot(X2D[:, 0][y==0], X2D[:, 1][y==0], "bs")
plt.plot(X2D[:, 0][y==1], X2D[:, 1][y==1], "g^")
plt.xlabel(r"$x_1$", fontsize=20)
plt.ylabel(r"$x_2$", fontsize=20, rotation=0)
plt.gca().get_yaxis().set_ticks([0, 4, 8, 12, 16])
plt.plot([-4.5, 4.5], [6.5, 6.5], "r--", linewidth=3)
plt.axis([-4.5, 4.5, -1, 17])
#调整两幅图的间距
plt.subplots_adjust(right=1)

plt.show()
```

![](C:\ML-Excercise\pictures\SVM\4.png)

如上面左边的情况，正方形和三角形对应两种类别的数据，在一维的情况下，无法找到一个超平面（在一维情况下是一个点）能将两种不同种类的数据分开。此时我们就需要加入更多的特征，一个方法是采用升维的方法，例如右边，增加了一个维度X^2，这样就可以在二维条件下用超平面（此情况为一条直线）将两种类别的数据很好的分开来。当然不止可以用X ^2，可以采用高次的多项式，如X ^2+X+1等也都可以尝试。在许多种机器学习方法中，都可以采用多项式来增加样本特征值，增维来实现样本更好的划分。

SVM里面采用**核函数**来升维，使得非线性分类问题转化为高维的线性分类问题。常见的核函数有线性核函数、多项式核函数、高斯核函数、Sigmoid核函数等，其形式如下：

![](C:\ML-Excercise\pictures\SVM\5.jpg)

下面以**多项式核函数**和**高斯核函数**为例展示一下SVM在非线性分类中的效果。

首先使用scikit-learn自带的moon数据，这个数据是专门用来实验分类算法的，其分布像是两个交错在一起的半圆。

```python
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)

plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.show()
```

![](C:\ML-Excercise\pictures\SVM\5.png)

下面使用**多项式核函数**来将该数据进行“升维”，这里进行了两种实验，一个多项式最高次为3，另一个最高次为10。

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
# 使用SVC来作为SVM分类器函数
from sklearn.svm import SVC

# 选择多项式核函数，最高项为3，常数项为1，惩罚参数为5
poly_kernel_svm_clf = Pipeline([
    	# 先标准化数据
        ("scaler", StandardScaler()),
        # 选择核函数为poly，即多项式核函数
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1,C=5))
    ])
poly_kernel_svm_clf.fit(X, y)

# 选择多项式核函数，最高项为10，常数项为100，惩罚参数为5
poly100_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=10, coef0=100, C=5))
    ])
poly100_kernel_svm_clf.fit(X, y)

```

画图

```python
def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    # 画出网格点
    x0, x1 = np.meshgrid(x0s, x1s)
    # x0.ravel()将多维数组降到一维
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    # 画出对网格点预测结果的等高线，alpha是透明度
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)

# 设置图幅的宽高
plt.figure(figsize=(11, 4))

plt.subplot(121)
plot_predictions(poly_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.title(r"$d=3, r=1, C=5$", fontsize=18)

plt.subplot(122)
plot_predictions(poly100_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.title(r"$d=10, r=100, C=5$", fontsize=18)

plt.show()
```

![](C:\ML-Excercise\pictures\SVM\6.png)

由上图可以看出，最高次项为10的多项式核函数分类结果比最高次项为3的要好。不过也要预防出现过拟合情况。

再试试使用**高斯核函数**来对数据进行分类。分别设置高斯核函数系数为0.1、0.5，惩罚系数C为0.001、1000.

```python
gamma1, gamma2 = 0.1, 5
C1, C2 = 0.001, 1000
hyperparams = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)

svm_clfs = []
for gamma, C in hyperparams:
    rbf_kernel_svm_clf = Pipeline([
            ("scaler", StandardScaler()),
            # 选择核函数为高斯核
            ("svm_clf", SVC(kernel="rbf", gamma=gamma, C=C))
        ])
    rbf_kernel_svm_clf.fit(X, y)
    svm_clfs.append(rbf_kernel_svm_clf)

plt.figure(figsize=(11, 7))

for i, svm_clf in enumerate(svm_clfs):
    plt.subplot(221 + i)
    plot_predictions(svm_clf, [-1.5, 2.5, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    gamma, C = hyperparams[i]
    plt.title(r"$\gamma = {}, C = {}$".format(gamma, C), fontsize=16)

#save_fig("moons_rbf_svc_plot")
plt.show()
```

![](C:\ML-Excercise\pictures\SVM\7.png)

