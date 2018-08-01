import numpy as np
import tensorflow as tf
from data_manager import DataManager


class Model:
    label_dim = 2
    # layer dims
    hidden_dim_1 = 21
    hidden_dim_2 = 10
    hidden_dim_3 = 5
    # lr
    learning_rate = 0.0001

    def __init__(self, input_dim):
        # place holders
        ph_inputs = tf.placeholder(tf.float32, [None, input_dim])
        ph_labels = tf.placeholder(tf.float32, [None, self.label_dim])

        # hidden 1
        with tf.name_scope("hidden1"):
            w_1 = tf.Variable(tf.truncated_normal([input_dim, self.hidden_dim_1], stddev=0.0001), name="w1")
            b_1 = tf.Variable(tf.zeros([self.hidden_dim_1]), name="b1")
            h_1 = tf.nn.relu(tf.matmul(ph_inputs, w_1) + b_1)
            tf.summary.histogram("w1", w_1)
        # hidden 2
        with tf.name_scope("hidden2"):
            w_2 = tf.Variable(tf.truncated_normal([self.hidden_dim_1, self.hidden_dim_2], stddev=0.0001), name="w2")
            b_2 = tf.Variable(tf.zeros([self.hidden_dim_2]), name="b2")
            h_2 = tf.nn.relu(tf.matmul(h_1, w_2) + b_2)
            tf.summary.histogram("w2", w_2)
        # hidden 3
        with tf.name_scope("hidden3"):
            w_3 = tf.Variable(tf.truncated_normal([self.hidden_dim_2, self.hidden_dim_3], stddev=0.0001), name="w3")
            b_3 = tf.Variable(tf.zeros([self.hidden_dim_3]), name="b3")
            h_3 = tf.nn.relu(tf.matmul(h_2, w_3) + b_3)
            tf.summary.histogram("w3", w_3)
        # output
        with tf.name_scope("output"):
            w_4 = tf.Variable(tf.truncated_normal([self.hidden_dim_3, 2], stddev=0.0001), name="w4")
            b_4 = tf.Variable(tf.zeros([self.label_dim]), name="b4")
            op_output = tf.nn.softmax(tf.matmul(h_3, w_4) + b_4)
            tf.summary.histogram("w4", w_4)
        # loss function
        with tf.name_scope("loss"):
            op_loss = tf.reduce_mean(-tf.reduce_sum(ph_labels * tf.log(op_output + 1e-5), axis=[1]))
            tf.summary.scalar("loss", op_loss)
        # train step
        with tf.name_scope("train"):
            op_train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(op_loss)
        # acc
        with tf.name_scope("accuracy"):
            op_correct = tf.equal(tf.arg_max(op_output, 1), tf.arg_max(ph_labels, 1))
            op_accuracy = tf.reduce_mean(tf.cast(op_correct, tf.float32))
            tf.summary.scalar("accuracy", op_accuracy)

        # set attributes
        self.ph_inputs = ph_inputs
        self.ph_labels = ph_labels
        self.op_output = op_output
        self.op_loss = op_loss
        self.op_train_step = op_train_step
        self.op_accuracy = op_accuracy


def main():
    dm = DataManager()
    train_x, train_y, val_x, val_y, test_x, test_y = dm.get_inputs(dim=7)
    model = Model(input_dim=train_x.shape[1])

    # prepare feed dicts
    val_feed_dict = {
        model.ph_inputs: val_x,
        model.ph_labels: val_y
    }
    test_feed_dict = {
        model.ph_inputs: test_x,
        model.ph_labels: test_y
    }

    # train nn model
    init = tf.global_variables_initializer()
    summary = tf.summary.merge_all()
    max_epoch = 30000
    n_batch = 256
    with tf.Session() as sess:
        sess.run(init)
        summary_writer = tf.summary.FileWriter("./logs", sess.graph)

        for step in range(max_epoch):
            batch = np.random.choice(len(train_x), n_batch, replace=False)
            train_x_batch = train_x[batch]
            train_y_batch = train_y[batch]
            sess.run(model.op_train_step, feed_dict={
                model.ph_inputs: train_x_batch,
                model.ph_labels: train_y_batch
            })

            if (step + 1) % 10 == 0:
                acc_val = sess.run(model.op_accuracy, feed_dict=val_feed_dict)
                loss_val = sess.run(model.op_loss, feed_dict=val_feed_dict)
                summary_str = sess.run(summary, feed_dict=val_feed_dict)
                summary_writer.add_summary(summary_str, step + 1)
                print("Step {0} accuracy: {1:.5f}, loss: {2:.5f}".format(step + 1, acc_val, loss_val))

        test_acc_val = sess.run(model.op_accuracy, feed_dict=test_feed_dict)
        print("\nTest Accuracy: {0}".format(test_acc_val))


if __name__ == "__main__":
    main()

