import tensorflow as tf
import numpy as np


def get_weight(shape, lamb):
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(lamb)(var))
    return var


x = tf.placeholder(dtype=tf.float32, shape=(None, 2))
y_ = tf.placeholder(dtype=tf.float32, shape=(None, 1))

batch_size=8

layer_dimension=[2,10,10,10,1]

layer_length = len(layer_dimension)

cur_layer=x
in_dimension= layer_dimension[0]

for i in range(1,layer_length):
    out_dimension = layer_dimension[i]
    Weight=get_weight([in_dimension,out_dimension],0.001)
    Bias = tf.Variable(tf.constant(0.1,tf.float32,shape=[out_dimension]))
    cur_layer = tf.nn.relu(tf.matmul(cur_layer,Weight)+Bias)
    in_dimension = layer_dimension[i]

mes_loss= tf.reduce_mean(tf.square(y_-cur_layer))

tf.add_to_collection("losses",mes_loss)

loss= tf.add_n(tf.get_collection("losses"))

dataSize_size = 128
X_input = np.random.rand(dataSize_size, 2)
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X_input]

with tf.Session() as sess:
    init_op= tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(loss,feed_dict={x:X_input,y_:Y}))
