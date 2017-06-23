from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

boston=load_boston()

X=boston.data
y=boston.target

ss_X=StandardScaler()
ss_y=StandardScaler()

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=33,test_size=0.25)

X_train=ss_X.fit_transform(X_train)
X_test=ss_X.transform(X_test)

y_train=ss_y.fit_transform(y_train)
y_test=ss_y.transform(y_test)

lr= LinearRegression()
lr.fit(X_train, y_train)
lr_y_predict=lr.predict(X_test)

sg= SGDRegressor()
sg.fit(X_train, y_train)
sg_y_predict=sg.predict(X_test)

print("linear regressor")
print(lr.score(X_test,y_test))
print(r2_score(y_test,lr_y_predict))
print(mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))
print(mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))

print("SGD regressor")
print(sg.score(X_test,y_test))
print(r2_score(y_test,sg_y_predict))
print(mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sg_y_predict)))
print(mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sg_y_predict)))


print("SVR")
from sklearn.svm import SVR

print("SVR--linear")
linear_svr=SVR(kernel='linear')
linear_svr.fit(X_train, y_train)
y_predict=linear_svr.predict(X_test)
print(linear_svr.score(X_test,y_test))
print(mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(y_predict)))
print(mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(y_predict)))

print("SVR--poly")
linear_ploy=SVR(kernel='poly')
linear_ploy.fit(X_train, y_train)
y_predict=linear_ploy.predict(X_test)
print(linear_ploy.score(X_test,y_test))
print(mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(y_predict)))
print(mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(y_predict)))

print("SVR--rbf")
linear_rbf=SVR(kernel='rbf')
linear_rbf.fit(X_train, y_train)
y_predict=linear_rbf.predict(X_test)
print(linear_rbf.score(X_test,y_test))
print(mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(y_predict)))
print(mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(y_predict)))