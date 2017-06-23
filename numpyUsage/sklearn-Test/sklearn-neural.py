from sklearn.neural_network import MLPClassifier
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(X, y)

print(clf.predict([[2., 2.], [-1., -2.]]))
print(clf.predict_proba([[2., 2.], [-1., -2.]]))
print(clf.coefs_)

[print(coef.shape) for coef in clf.coefs_]

import math
z = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
z_exp = [math.exp(i) for i in z]
print(z_exp)  # Result: [2.72, 7.39, 20.09, 54.6, 2.72, 7.39, 20.09]
sum_z_exp = sum(z_exp)
print(sum_z_exp)  # Result: 114.98
softmax = [round(i / sum_z_exp, 3) for i in z_exp]
print(max(softmax))  # Result: [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]