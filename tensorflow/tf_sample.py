import tensorflow as tf
import numpy as np

batch_size = 8

W1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
W2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

a = tf.matmul(x, W1)
y = tf.matmul(a, W2)

cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

dataSize_size = 128
X_input = np.random.rand(dataSize_size, 2)
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X_input]


with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(W1))
    print(sess.run(W2))

    STEPS=1000
    for i in range(STEPS):
        start= (i * batch_size) % dataSize_size
        end = min(start+batch_size, dataSize_size)
        sess.run(train_step, feed_dict={x:X_input[start:end],y_:Y[start:end]})

        if i%100 ==0:
            total_cross_entropy=sess.run(cross_entropy,feed_dict={x:X_input,y_:Y})
            print(i,total_cross_entropy)

    print(sess.run(W1))
    print(sess.run(W2))