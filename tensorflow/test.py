import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

c1 = np.array([1, 2, 3]).reshape((1,3,1))
c2 = np.array([-1, -2, -3, -4, -5]).reshape((1,1,5))
print((c1,c2))

sess = tf.InteractiveSession()
x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])
x.initializer.run()
sub = tf.sub(x, a)
print(sub.eval())

rd= np.linspace(-1,1,100)
print(rd[:,np.newaxis].shape)

aa = np.arange(0,10).reshape(1,10)
print(aa)
print(sess.run(tf.arg_max(aa,1)))


mnist= input_data.read_data_sets("",one_hot=True)
print(mnist.train.labels[0:2])

