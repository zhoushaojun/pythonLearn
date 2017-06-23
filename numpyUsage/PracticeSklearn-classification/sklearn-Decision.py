import pandas as pd
from sklearn.feature_extraction import  DictVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import numpy as np

titanic=pd.read_excel('C:\\Users\\zhoushaojun\\Desktop\\titanic.xls')
#titanic.head()
print(titanic.shape)
print(type(titanic))
#print(titanic.head())
X=titanic[['pclass','age','sex']]
y=titanic['survived']
print(X.info())

X['age'].fillna(X['age'].mean(),inplace=True)
print(X.info())

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.23,random_state=33)
dicVec=DictVectorizer(sparse=False)
print(X_train)
print(X_train.to_dict(orient="record"))
X_train=dicVec.fit_transform(X_train.to_dict(orient="record"))
print(X_train)
print(dicVec.get_feature_names())
X_test=dicVec.transform(X_test.to_dict(orient="record"))

#单一决策
dc=DecisionTreeClassifier(criterion='entropy')
dc.fit(X_train,y_train)
y_predict=dc.predict(X_test)

print("Decision Tree")
print(dc.score(X_test,y_test))
print(classification_report(y_test,y_predict))

'''
#随机森林
from sklearn.ensemble import RandomForestClassifier
dc=RandomForestClassifier()
dc.fit(X_train,y_train)
ry_predict=dc.predict(X_test)

print("Random Decision Tree")
print(dc.score(X_test,y_test))
print(classification_report(y_test,ry_predict))

#随机森林
from sklearn.ensemble import GradientBoostingClassifier
dc=GradientBoostingClassifier()
dc.fit(X_train,y_train)
gy_predict=dc.predict(X_test)

print("Gradient Decision Tree")
print(dc.score(X_test,y_test))
print(classification_report(y_test,gy_predict))
'''

from sklearn.cross_validation import cross_val_score

x = np.arange(9.)
print(x==7)
print(np.where( x ==7 ))