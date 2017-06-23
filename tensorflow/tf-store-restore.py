import tensorflow as tf

''' 普通的方式加载
v1=tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
v2=tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
result=v1+v2

#init_op=tf.global_variables_initializer()
saver=tf.train.Saver()

with tf.Session() as sess:
    #sess.run(init_op)
    #saver.save(sess,'./models/model.ckpt')
    saver.restore(sess,'./models/model.ckpt')
    print(sess.run(result))

'''

''' 指数方式'''

''' 保存
v = tf.Variable(0, dtype=tf.float32, name="v")

for variables in tf.global_variables():
    print(variables.name)

ema= tf.train.ExponentialMovingAverage(0.99)
ema_op = ema.apply(tf.global_variables())


for variables in tf.global_variables():
    print(variables.name)

saver= tf.train.Saver()

with tf.Session() as sess:
    init_op= tf.global_variables_initializer()
    sess.run(init_op)

    sess.run(tf.assign(v,10))
    sess.run(ema_op)

    saver.save(sess,"./models/ema.ckpt")
    print(sess.run([v,ema.average(v)]))
'''

'''读取
v = tf.Variable(0, dtype=tf.float32, name="v")

ema= tf.train.ExponentialMovingAverage(0.99)

print(ema.variables_to_restore())

saver= tf.train.Saver(ema.variables_to_restore())
with tf.Session() as sess:
    saver.restore(sess,"./models/ema.ckpt")
    print(sess.run(v))
'''

# 图的方式 保存
'''
from tensorflow.python.framework import graph_util

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')

result = v1 + v2

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    graph_def = tf.get_default_graph().as_graph_def()
    out_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])
    with tf.gfile.GFile("./models/conbined_model.pb","wb") as f:
        f.write(out_graph_def.SerializeToString())
'''

#图的方式 读取
from tensorflow.python.platform import gfile

with tf.Session() as sess:
    with tf.gfile.FastGFile("./models/conbined_model.pb") as f:
        graph_def = tf.get_default_graph().as_graph_def()
        graph_def.ParseFromString(f.read())
    result= tf.import_graph_def(graph_def, return_elements=["add:0"])
    print(sess.run(result))
