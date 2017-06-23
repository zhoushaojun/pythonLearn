from sklearn.datasets import load_iris
from sklearn import feature_selection as fs
from sklearn.feature_selection import chi2
iris = load_iris()
X, y = iris.data, iris.target
print(X.shape)

X_new = fs.SelectKBest(fs.chi2, k=2).fit_transform(X, y)
print(X_new.shape)

X_new = fs.SelectPercentile(fs.chi2, percentile=20).fit_transform(X, y)
print(X_new.shape)