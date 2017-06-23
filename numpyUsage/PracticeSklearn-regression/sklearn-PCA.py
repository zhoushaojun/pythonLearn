import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

M = np.array([[1, 2], [2, 4]])
print(np.linalg.matrix_rank(M))

digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra')

digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes')

X_digits = digits_train[np.arange(64)]
y_digits = digits_train[[64]]

estimator = PCA(n_components=2)
X_pca = estimator.fit_transform(X_digits)

def plot_pca_scatter():
    colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
    for i in range(len(colors)):
        px = X_pca[:, 0:1][y_digits.as_matrix() == i]
        py = X_pca[:, 1:2][y_digits.as_matrix() == i]
        plt.scatter(px,py,c=colors[i])
    plt.legend(np.arange(0,10).astype(str))
    plt.xlabel("px")
    plt.ylabel("py")
    plt.show()

plot_pca_scatter()








