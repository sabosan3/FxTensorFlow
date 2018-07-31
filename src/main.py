# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import tensorflow as tf

if __name__=="__main__":
    
    ### fx types
    FILES = [
            "USDJPY",
            "AUDJPY",
            "EURJPY",
            "GBPJPY",
            "NZDJPY",
            "EURUSD"
            ]
    
    ### for rename cloumns
    COLUMNS = {
            "始値": "open",
            "高値": "max",
            "安値": "min",
            "終値": "close"
            }
    
    file_path = os.path.join("..", "data")
    
    target = "USDJPY-close"
    
    all_df = []
    for file in FILES:
        
        ### create new columns name
        NEW_COLUMNS = {i:file + "-" + j for i,j in COLUMNS.items()}
        
        data_file = os.path.join(file_path, file + ".csv")
        
        ### load fx data from csv file
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        df.rename(columns=NEW_COLUMNS, inplace=True)
        df.index.rename("date", inplace=True)
        
        ### obtain new columns name
        open_str, max_str, min_str, close_str = df.columns
        
        all_df.append(df[close_str])

    ### format all types of fx closing values to pandas DataFrame
    all_df = pd.concat(all_df, axis=1)
    
    ### for normalizing all values
    normalize_func = lambda x: x / np.max(x)
    all_df = all_df.apply(normalize_func)
    
    ### log difference
    all_df = np.log(all_df / all_df.shift())
    all_df.dropna(inplace=True)
    
    ### calculate differences between previous and today closing value
    diff_target = all_df[target].diff()
    diff_target.dropna(inplace=True)
    
    ### make one hot labels
    labels = diff_target > 0
    labels = pd.concat([labels, ~labels], axis=1)
    labels = labels.astype(float)
    all_df = all_df.loc[labels.index]

    ### separate training and test data
    train_df = all_df["2007":"2017"]
    test_df = all_df["2018"]

    ### calculate delay data for all
    def calc_delay_data_all(df, dim):
        all_data = {}
        for col, values in df.iteritems():
            all_data[col] = calc_delay_data(values, dim)
        all_delay_data = pd.Panel(all_data)
        return all_delay_data
    
    ### calculate delay data
    def calc_delay_data(values, dim):
        columns = ["before {0} day".format(i) for i in range(1, dim + 1)]
        indexes = []
        delay_data = []
        for i in range(len(values)):
            if i - dim < 0:
                 continue
            indexes.append(values.index[i])
            delay = []
            for d in range(1, dim + 1):
                delay.append(values.iloc[i - d])
            delay_data.append(delay)
        delay_data = pd.DataFrame(delay_data, index=indexes, columns=columns)
        return delay_data
    
    ### calculate training data 
    dim = 7
    train_delay = calc_delay_data_all(train_df, dim)

    tmp = []
    for s in all_df.columns:
       tmp.append(train_delay[s])
       
    train_delay = pd.concat(tmp, axis=1)
    
    ### calculate test data
    test_delay = calc_delay_data_all(test_df, dim)

    tmp = []
    for s in all_df.columns:
       tmp.append(test_delay[s])
       
    test_delay = pd.concat(tmp, axis=1)    
        
    ### define nn model
    
    # input
    input = tf.placeholder(tf.float32, [None, 42])
    
    # hidden 1
    with tf.name_scope("hidden1"):
        w_1 = tf.Variable(tf.truncated_normal([42, 21], stddev=0.0001), name="w1")
        b_1 = tf.Variable(tf.zeros([21]), name="b1")
        h_1 = tf.nn.relu(tf.matmul(input, w_1) + b_1)
        tf.summary.histogram("w1", w_1)
        
    # hidden 2
    with tf.name_scope("hidden2"):
        w_2 = tf.Variable(tf.truncated_normal([21, 10], stddev=0.0001), name="w2")
        b_2 = tf.Variable(tf.zeros([10]), name="b2")
        h_2 = tf.nn.relu(tf.matmul(h_1, w_2) + b_2)
        tf.summary.histogram("w2", w_2)
        
    # hidden 3
    with tf.name_scope("hidden3"):
        w_3 = tf.Variable(tf.truncated_normal([10, 5], stddev=0.0001), name="w3")
        b_3 = tf.Variable(tf.zeros([5]), name="b3")
        h_3 = tf.nn.relu(tf.matmul(h_2, w_3) + b_3)
        tf.summary.histogram("w3", w_3)
        
    # output
    with tf.name_scope("output"):
        w_4 = tf.Variable(tf.truncated_normal([5, 2], stddev=0.0001), name="w4")
        b_4 = tf.Variable(tf.zeros([2]), name="b4")
        output = tf.nn.softmax(tf.matmul(h_3, w_4) + b_4)
        tf.summary.histogram("w4", w_4)

    # loss function
    ground_truth = tf.placeholder(tf.float32, [None, 2])
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(-tf.reduce_sum(ground_truth * tf.log(output + 1e-5), axis=[1]))
        tf.summary.scalar("loss", loss)
    
    # train step
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
           
    # evaluation
    with tf.name_scope("accuracy"):
        correct = tf.equal(tf.arg_max(output, 1), tf.arg_max(ground_truth, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    # init variables        
    init = tf.global_variables_initializer()
    summary = tf.summary.merge_all()
        
    ### train nn model

    # separate validate data
    train_x = train_delay["2007":"2016"]
    train_y = labels.loc[train_x.index]
    
    validata_x = train_delay["2017"]
    validata_y = labels.loc[validata_x.index]
    validata_feed_dict = {
            input: validata_x.as_matrix(),
            ground_truth: validata_y.as_matrix()
            }
    
    test_x = test_delay
    test_y = labels.loc[test_x.index]
    test_feed_dict = {
            input: test_x.as_matrix(),
            ground_truth: test_y.as_matrix()
            }

    with tf.Session() as sess:
        
        sess.run(init)
        summary_writer = tf.summary.FileWriter(".\\logs", sess.graph)

        for step in range(30000):            
            batch = np.random.choice(len(train_x), 256, replace=False)
            train_x_batch = train_x.as_matrix()[batch]
            train_y_batch = train_y.as_matrix()[batch]
            train_feed_dict = {
                    input: train_x_batch,
                    ground_truth: train_y_batch
                    }
            
            sess.run(train_step, feed_dict=train_feed_dict)
            
            if (step + 1) % 10 == 0:
                acc_val = sess.run(accuracy, feed_dict=validata_feed_dict)
                loss_val = sess.run(loss, feed_dict=validata_feed_dict)
                summary_str = sess.run(summary, feed_dict=validata_feed_dict)
                summary_writer.add_summary(summary_str, step + 1)
                print("Step {0} accuracy: {1:.5f}, loss: {2:.5f}".format(step+1, acc_val, loss_val))
        
        test_acc_val = sess.run(accuracy, feed_dict=test_feed_dict)        
        print()
        print("Test Accuracy: {0}".format(test_acc_val))