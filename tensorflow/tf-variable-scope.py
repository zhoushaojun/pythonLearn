import tensorflow as tf


def variableScope(reuse):
    with tf.variable_scope('lay1',reuse=reuse):
       # W1 = tf.Variable(tf.truncated_normal([1, 10], stddev=1.0),name='w1')
       W1= tf.get_variable("w1",initializer=tf.truncated_normal(shape=(1,10),stddev=1.0))
       print(W1.name)
    return W1


def main(argv=None):
    B1 =variableScope(False)
    B2 =variableScope(True)
    with tf.Session() as sess:
        all_op = tf.global_variables_initializer()
        sess.run(all_op)
        print(sess.run(B1))
        print(sess.run(B2))

if __name__ == '__main__':
    tf.app.run()
