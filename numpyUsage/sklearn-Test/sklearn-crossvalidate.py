from sklearn import datasets, svm
import  numpy as np
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
svc = svm.SVC(C=1, kernel='linear')
sfit = svc.fit(X_digits[:-100], y_digits[:-100])
score= sfit.score(X_digits[-100:], y_digits[-100:])
print(score)

#手写cross验证
'''
import numpy as np
X_folds = np.array_split(X_digits, 3)
y_folds = np.array_split(y_digits, 3)
scores = list()
for k in range(3):
    # We use 'list' to copy, in order to 'pop' later on
    X_train = list(X_folds)
    X_test  = X_train.pop(k)
    X_train = np.concatenate(X_train)
    y_train = list(y_folds)
    y_test  = y_train.pop(k)
    y_train = np.concatenate(y_train)
    scores.append(svc.fit(X_train, y_train).score(X_test, y_test))
print(scores)
'''

#cross验证器
'''
from sklearn import cross_validation
k_fold = cross_validation.KFold(n=6, n_folds=3)
for train_indices, test_indices in k_fold:
     print('Train: %s | test: %s' % (train_indices, test_indices))

kfold = cross_validation.KFold(len(X_digits), n_folds=3)
print([svc.fit(X_digits[train], y_digits[train]).score(X_digits[test], y_digits[test])
'''

#网格搜索
from sklearn import svm, grid_search, datasets
iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(iris.data, iris.target)
#print(clf.best_score_)
#print(clf.best_estimator_.C)


#GridSearch搜索
from sklearn.grid_search import GridSearchCV
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
print(X_digits.shape)
svc = svm.SVC()
Cs = np.logspace(-6, -1, 10)
print(Cs)
clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs),
                   n_jobs=1)
clf.fit(X_digits[:-100], y_digits[:-100])
print(clf.best_score_)
print(clf.best_estimator_.C)
# Prediction performance on test set is not as good as on train set
print(clf.score(X_digits[-100:], y_digits[-100:]))


#同步验证 clf 为网格验证， cross_val_score同时度量模型的表现
from sklearn import cross_validation
cross_validation.cross_val_score(clf, X_digits, y_digits)
print(clf.best_score_)
print(clf.best_estimator_.C)
# Prediction performance on test set is not as good as on train set
print(clf.score(X_digits[-100:], y_digits[-100:]))