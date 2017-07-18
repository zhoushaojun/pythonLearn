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
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(Hidden_Size)
        if is_training:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=Keep_Prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * Num_Layer)

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
        output = tf.reshape(tf.concat(1, outputs), [-1, Hidden_Size])

        Weights = tf.get_variable("weight", [Hidden_Size, Vacab_Size])
        bias = tf.get_variable("bias", [Vacab_Size])
        logits = tf.matmul(output, Weights) + bias

        loss = tf.nn.seq2seq.sequence_loss_by_example([logits],
                                                      [tf.reshape(self.targets, [-1])],
                                                      tf.ones([batch_size * num_steps], dtype=tf.float32))
        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state

        if not is_training:
            return

        trainable_varibales = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_varibales), Max_Grad_Norm)

        optimizer = tf.train.GradientDescentOptimizer(L_Rate)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_varibales))

def run_epoch(session, model, data, train_op,output_log):
    total_costs = 0.0
    iters=0
    state = session.run(model.initial_state)

    for (x,y) in reader.ptb_iterator())