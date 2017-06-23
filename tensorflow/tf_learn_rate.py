import tensorflow as tf

global_step= tf.Variable(0)
learning_rate= tf.train.exponential_decay(0.1,global_step,100,0.96, staircase=True)

with tf.Session() as sess:
    init_op= tf.global_variables_initializer()
    sess.run(init_op)
    global_step = global_step.assign(200)
    print(sess.run(global_step))
    print(sess.run(learning_rate))