from tensorflow.models.rnn.ptb import reader
import tensorflow as tf
import numpy as np

DATA_PATH = ".\data\data"
Hidden_Size = 200
Num_Layer = 2
Vacab_Size = 10000

L_Rate = 1.0
Train_Batch_Size = 20
Train_Num_Step = 35

Eval_Batch_Size = 1
Eval_Num_Step = 1
Num_Epoch = 1
Keep_Prob = 0.5
Max_Grad_Norm = 5


class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        self.bath_size = batch_size
        self.num_steps = num_steps
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(Hidden_Size)
        if is_training:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=Keep_Prob)
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * Num_Layer)

        self.initial_state = cell.zero_state(batch_size, tf.float32)
        embedding = tf.get_variable("embedding", [Vacab_Size, Hidden_Size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        if is_training:
            inputs = tf.nn.dropout(inputs, Keep_Prob)
        outputs = []
        state = self.initial_state

        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
                print(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, Hidden_Size])

        Weights = tf.get_variable("weight", [Hidden_Size, Vacab_Size])
        bias = tf.get_variable("bias", [Vacab_Size])
        logits = tf.matmul(output, Weights) + bias
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],
                                                                  [tf.reshape(self.targets, [-1])],
                                                                  [tf.ones([batch_size * num_steps], dtype=tf.float32)])
        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state

        if not is_training:
            return

        trainable_varibales = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_varibales), Max_Grad_Norm)

        optimizer = tf.train.GradientDescentOptimizer(L_Rate)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_varibales))


def run_epoch(session, model, data, train_op, output_log, epoch_size):
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    for step in range(epoch_size):
        x, y = session.run(data)
        cost, state, _ = session.run([model.cost, model.final_state, train_op], feed_dict={
            model.input_data: x, model.targets: y, model.initial_state: state
        })

        total_costs += cost
        iters += model.num_steps
        if output_log and step % 100 == 0:
            print("After %d steps, perplexity is %.3f" % (step, np.exp(total_costs / iters)))
    return np.exp(total_costs / iters)


def main():
    train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)
    # 计算一个epoch需要训练的次数"
    train_data_len = len(train_data)
    train_batch_len = train_data_len // Train_Batch_Size
    train_epoch_size = (train_batch_len - 1) // Train_Num_Step

    valid_data_len = len(valid_data)
    valid_batch_len = valid_data_len // Eval_Batch_Size
    valid_epoch_size = (valid_batch_len - 1) // Eval_Num_Step

    test_data_len = len(test_data)
    test_batch_len = test_data_len // Eval_Batch_Size
    test_epoch_size = (test_batch_len - 1) // Eval_Num_Step

    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    with tf.variable_scope("language_model", reuse=None, initializer=initializer):
        train_model = PTBModel(True, Train_Batch_Size, Train_Num_Step)
    with tf.variable_scope("language_model", reuse=True, initializer=initializer):
        eval_model = PTBModel(False, Eval_Batch_Size, Eval_Num_Step)
    # 训练模型。

    with tf.Session() as session:
        tf.global_variables_initializer().run()

        train_queue = reader.ptb_producer(train_data, Train_Batch_Size, Train_Num_Step)
        eval_queue = reader.ptb_producer(valid_data, Eval_Batch_Size, Eval_Num_Step)
        test_queue = reader.ptb_producer(test_data, Eval_Batch_Size, Eval_Num_Step)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)

        for i in range(Num_Epoch):
            print(i)
            print("In iteration: %d" % (i + 1))
            run_epoch(session, train_model, train_queue, train_model.train_op, True, train_epoch_size)

            valid_perplexity = run_epoch(session, eval_model, eval_queue, tf.no_op(), False, valid_epoch_size)
            print("Epoch: %d Validation Perplexity: %.3f" % (i + 1, valid_perplexity))

        test_perplexity = run_epoch(session, eval_model, test_queue, tf.no_op(), False, test_epoch_size)
        print("Test Perplexity: %.3f" % test_perplexity)

        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    main()
