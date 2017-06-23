import numpy as np
X = [1,2]
state=[.0,.0]

w_cell_state=np.asarray([[0.1,0.2],[0.3,0.4]])
w_cell_input=np.asarray([0.5,0.6])

w_output=np.asarray([[1.0],[2.0]])
b_output =0.1

for i in range(len(X)):
    before_activation=np.dot(state, w_cell_state) + X[i]*w_cell_input
    state = np.tanh(before_activation)

    final_output=np.dot(state,w_output)+b_output

    print(before_activation)
    print(state)
    print(final_output)