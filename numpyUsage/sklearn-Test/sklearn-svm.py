#SVM

#分类
'''
from sklearn import svm
import numpy as np
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)

ret = clf.predict([[2., 2.]])
print(ret)
print(clf.support_vectors_)

print("ovo svc")
X = [[0,1], [1,2], [2,3], [3,4]]
Y = [0, 1,1,0]
clf = svm.SVC(kernel="linear",decision_function_shape='ovo')
clf.fit(X, Y)
print(clf.coef_)
print(clf.intercept_)

print("liner svc")
lin_clf = svm.LinearSVC()
lin_clf.fit(X, Y)
print(lin_clf.coef_)
print(lin_clf.intercept_)

print("liner kenel svc")
lin_clf = svm.SVC(kernel="linear")
lin_clf.fit(X, Y)
print(lin_clf.coef_)
print(lin_clf.intercept_)
'''

#线性分类器 画图
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
#from sklearn.linear_model import SGDClassifier

# we create 40 separable points
# we create 40 separable points
np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20

# fit the model
clf = svm.SVC(kernel='linear',probability=True)
clf.fit(X, Y)

# get the separating hyperplane
w = clf.coef_[0]
a = -w[1]
print(w,a)
print(clf.coef_)
print(clf.intercept_)
print(clf.predict_proba)
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0])
# plot the parallels to the separating hyperplane that pass through the
# support vectors
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

# plot the line, the points, and the nearest vectors to the plane
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=80, facecolors='none')
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

plt.axis('tight')
plt.show()
'''

#回归

from sklearn import svm
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
clf = svm.SVR()
clf.fit(X, y)
svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
print(clf.predict([[1, 1]]))

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
print(X.shape)
y = iris.target
print(y.shape)
X = X[y != 0, :2]
y = y[y != 0]
print(X.shape)