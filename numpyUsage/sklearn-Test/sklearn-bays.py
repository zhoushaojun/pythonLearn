from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
gnb = MultinomialNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print(np.unique(iris.target))
print(y_pred.shape)
print("Number of mislabeled points out of a total %d points : %d"
      % (iris.data.shape[0],(iris.target != y_pred).sum()))