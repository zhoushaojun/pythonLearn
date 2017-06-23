from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import linear_model

'''
iris = datasets.load_iris()
data = iris.data
print(data.shape)

digits = datasets.load_digits()
print(digits.images.shape)
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r)
# plt.show()
print(digits.images.shape[0])
data = digits.images.reshape((digits.images.shape[0], 64))
print(data.shape)
'''

'''
# (一)近邻高维 KNN
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
np.unique(iris_y)

np.random.seed(0)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train)
print(knn.predict(iris_X_test))
print(iris_y_test)

# 计算就近的近邻
from sklearn.neighbors import NearestNeighbors
import numpy as np

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
print(indices)
print(distances)
print(nbrs.kneighbors_graph(X).toarray())

# 计算点点关系， <r[i]统计
from sklearn.neighbors import KDTree
import numpy as np

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
kdt = KDTree(X, leaf_size=30, metric='euclidean')
d, i = kdt.query(X[5], k=2, return_distance=True)

X = np.random.random((10, 5))
r = np.linspace(0, 1, 5)
tree = KDTree(X)
# doctest: +SKIP
print(X, r)

ret = tree.two_point_correlation([2, 2, 2, 2, 2], r)
print(ret)
'''

print("linermodel start-----")
diabetes = datasets.load_diabetes()
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test = diabetes.data[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

#(二)线性模型
import pylab as pl
'''
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)
print(regr.coef_)
# The mean square error
print(np.mean((regr.predict(diabetes_X_test)-diabetes_y_test)**2))
# Explained variance score: 1 is perfect prediction
# and 0 means that there is no linear relationship
# between X and Y.
print(regr.score(diabetes_X_test, diabetes_y_test))
print(regr.predict(diabetes_X_test))
print(diabetes_y_test)
'''


'''
#三种线性模型
reg = linear_model.LinearRegression()
reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
print(reg.coef_)

from sklearn import linear_model
reg = linear_model.Ridge (alpha = .5)
reg.fit ([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
print(reg.coef_)

reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
print(reg.predict([2,2]))
print(reg.coef_)
print(reg.alpha_)

X = np.c_[ .5, 1].T
y = np.c_[.5, 1].T
test = np.c_[ 0, 2].T

'''

'''
# 普通线性模型
regr = linear_model.LinearRegression()
pl.figure()
np.random.seed(0)
for _ in range(6):
   this_X = .1*np.random.normal(size=(2, 1)) + X
   regr.fit(this_X, y)
   pl.plot(test, regr.predict(test))
   pl.scatter(this_X, y, s=3)
pl.show()
'''

'''
#岭回归
regr = linear_model.Ridge(alpha=.1)
pl.figure()
np.random.seed(0)
for _ in range(6):
   this_X = .1*np.random.normal(size=(2, 1)) + X
   regr.fit(this_X, y)
   pl.plot(test, regr.predict(test))
   pl.scatter(this_X, y, s=3)
pl.show()
'''

'''
#lasso回归
regr = linear_model.Ridge(alpha=.1)
alphas = np.logspace(-4, -1, 6)
print([regr.set_params(alpha=alpha
            ).fit(diabetes_X_train, diabetes_y_train,
            ).score(diabetes_X_test, diabetes_y_test) for alpha in alphas])

print(regr.score(diabetes_X_test,diabetes_y_test))


regr = linear_model.Lasso()
scores = [regr.set_params(alpha=alpha
            ).fit(diabetes_X_train, diabetes_y_train
            ).score(diabetes_X_test, diabetes_y_test)
       for alpha in alphas]


best_alpha = alphas[scores.index(max(scores))]
regr.alpha = best_alpha
print(best_alpha)
regr.fit(diabetes_X_train, diabetes_y_train)

print(regr.score(diabetes_X_test,diabetes_y_test))
'''


#logistic逻辑四弟

from sklearn import datasets, neighbors, linear_model
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

from sklearn import datasets, neighbors, linear_model
import math
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
print(np.unique(y_digits))
n_samples = len(X_digits)
print(math.ceil(.9 * n_samples))
print(X_digits.shape)
X_train = X_digits[:math.ceil(.9 * n_samples)]
y_train = y_digits[:math.ceil(.9 * n_samples)]
X_test = X_digits[math.ceil(.9 * n_samples):]
y_test = y_digits[math.ceil(.9 * n_samples):]
knn = neighbors.KNeighborsClassifier()
logistic = linear_model.LogisticRegression()
logistic.fit(X_train, y_train)
print(logistic.predict(X_test))
print('KNN score: %f' % knn.fit(X_train, y_train).score(X_test, y_test))
print('LogisticRegression score: %f'
      % logistic.fit(X_train, y_train).score(X_test, y_test))


#线性模型 +画图 使用某一种特征
'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# Load the diabetes dataset
diabetes = datasets.load_diabetes()


# Use only one feature0
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
'''