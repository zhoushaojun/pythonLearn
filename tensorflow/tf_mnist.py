import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10

LAYER1_NODE = 500

BATCH_SIZE = 100

L_R = 0.8
L_R_D = 0.99

REGULAR_RATE = 1e-4
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99


def interface(input, W1, B1, W2, B2, avg_moving):
    if avg_moving == None:
        hide1_out = tf.nn.relu(tf.matmul(input, W1) + B1)
        return tf.matmul(hide1_out, W2) + B2
    else:
        hide1_out = tf.nn.relu(tf.matmul(input, avg_moving.average(W1)) + avg_moving.average(B1))
        return tf.matmul(hide1_out, avg_moving.average(W2)) + avg_moving.average(B2)


def train(mnist):
    INPUT_X = tf.placeholder(tf.float32, shape=(None, INPUT_NODE), name='x-input')
    INPUT_Y = tf.placeholder(tf.float32, shape=(None, OUTPUT_NODE), name='y-input')

    W1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=1.0))
    B1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    W2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=1.0))
    B2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    predict_y = interface(INPUT_X, W1, B1, W2, B2, None)

    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = interface(INPUT_X, W1, B1, W2, B2, variable_averages)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(predict_y,INPUT_Y)
    #或者采用稀疏方式
    #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(predict_y, tf.arg_max(INPUT_Y,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizer = tf.contrib.layers.l2_regularizer(REGULAR_RATE)
    regularizer_ret = regularizer(W1) + regularizer(W2)
    loss = cross_entropy_mean + regularizer_ret

    learning_rate = tf.train.exponential_decay(L_R, global_step, mnist.train.num_examples / BATCH_SIZE, L_R_D)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)
    train_op = tf.group(train_step, variable_averages_op)

    correct_prediction = tf.equal(tf.arg_max(average_y, 1), tf.arg_max(predict_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        all_op = tf.global_variables_initializer()
        sess.run(all_op)

        validate_feed = {INPUT_X: mnist.validation.images, INPUT_Y: mnist.validation.labels}
        test_feed = {INPUT_X: mnist.test.images, INPUT_Y: mnist.test.labels}

        for i in range(TRAINING_STEPS):
            if (i % 1000 == 0):
                validate_cc = sess.run(accuracy, feed_dict=validate_feed)
                print("training--curracy--%i--%g", i, validate_cc)
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            train_feed = {INPUT_X: xs, INPUT_Y: ys}
            sess.run(train_op,feed_dict=train_feed)

        # 最后测试数据
        test_cc = sess.run(accuracy, feed_dict=test_feed)
        print("last--curracy--%i--%g", TRAINING_STEPS, test_cc)


def main(argv=None):
    mnist = input_data.read_data_sets("", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
