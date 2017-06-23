from sklearn.preprocessing import  PolynomialFeatures
from sklearn.grid_search import GridSearchCV

X=[[6],[8],[10],[14],[18]]
Y=[[7],[9],[13],[17.5],[18]]

poly2=PolynomialFeatures(degree=2)
print(poly2.fit_transform(X))