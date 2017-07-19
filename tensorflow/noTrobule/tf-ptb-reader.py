from tensorflow.models.rnn.ptb import reader
import tensorflow as tf
import numpy as np


def testPtbProducer():
    train_data, valid_data, test_data, _ = reader.ptb_raw_data(".\data\data")
    raw_data = train_data
    batch_size = 3
    num_steps = 2
    x, y = reader.ptb_producer(raw_data, batch_size, num_steps)
    with tf.Session() as session:
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(session, coord=coord)
        try:
            for i in range(10):
                xval, yval = session.run([x, y])
                print(xval, yval)
        finally:
            coord.request_stop()
            coord.join()

testPtbProducer()