import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniform of Cell Shape',
                  'Marginal Adhesion',
                  'Single Epithelial Cell Size', 'Bare Nuclel', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses',
                  'Class']

data= pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
                  names=column_names)

data=data.replace(to_replace='? ', value=np.nan)
data=data.dropna(how='any')
print(data.shape)

X_train, X_test, y_tain, y_test =train_test_split(data[column_names[1:10]],data[column_names[10]], test_size=0.25, random_state=33)

#print(column_names[1:10])
print(y_tain.value_counts())
print(y_test.value_counts())
