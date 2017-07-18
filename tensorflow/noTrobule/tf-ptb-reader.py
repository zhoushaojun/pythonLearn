from tensorflow.models.rnn.ptb import reader
import tensorflow as tf
import numpy as np

train_data, valid_data,test_data,_ = reader.ptb_raw_data(".\data\data")



op =tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(op)
    for  (x,y)  in enumerate(reader.ptb_producer(train_data, 10, 10)):
        sess.run(x,y)
        print(x,y)