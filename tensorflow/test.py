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

B1 = tf.Variable(tf.constant(0.1, shape=[10]))
B2 = tf.Variable(tf.constant(0.2, shape=[2,10]))
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(B1))
    print(sess.run(B2))
    print(sess.run(B1+B2))

mnist= input_data.read_data_sets("",one_hot=True)
print(mnist.train.labels[0:2])


from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

# load data
digits = load_digits()
X = digits.data
y = digits.target
print(y[0:10])
y = LabelBinarizer().fit_transform(y)
print(y[0:10])

