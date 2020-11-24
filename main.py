# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 11:16:30 2020

@author: LSH
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import time

from utils import DKT
from load_data import DKTData

rnn_cells = {
    "LSTM": tf.contrib.rnn.LSTMCell,
    "GRU": tf.contrib.rnn.GRUCell,
    "BasicRNN": tf.contrib.rnn.BasicRNNCell,
    "LayerNormBasicLSTM": tf.contrib.rnn.LayerNormBasicLSTMCell,
}

network_config = {}
network_config['batch_size'] = 64
network_config['hidden_layer_structure'] = [200, ]
network_config['learning_rate'] = 1e-2
network_config['keep_prob'] = 0.5
network_config['rnn_cell'] = rnn_cells['LSTM']
network_config['max_grad_norm'] = 5.0
network_config['lambda_w1'] = 0.003
network_config['lambda_w2'] = 3.0
network_config['lambda_o'] = 0.1
network_config['embedding'] = False

num_runs = 5
num_epochs = 30
batch_size = 64
keep_prob = 0.5

dataset = 'a2009u'
if dataset == 'a2009u':
    train_path = './data/assist2009_updated/assist2009_updated_train.csv'
    test_path = './data/assist2009_updated/assist2009_updated_test.csv'
    save_dir_prefix = './results/a2009u/'
elif dataset == 'a2015':
    train_path = './data/assist2015/assist2015_train.csv'
    test_path = './data/assist2015/assist2015_test.csv'
    save_dir_prefix = './results/a2015/'
elif dataset == 'synthetic':
    train_path = './data/synthetic/naive_c5_q50_s4000_v1_train.csv'
    test_path = './data/synthetic/naive_c5_q50_s4000_v1_test.csv'
    save_dir_prefix = './results/synthetic/'
elif dataset == 'statics':
    train_path = './data/STATICS/STATICS_train.csv'
    test_path = './data/STATICS/STATICS_test.csv'
    save_dir_prefix = './results/STATICS/'
elif dataset =='assistment_challenge':
    train_path = './data/assistment_challenge/assistment_challenge_train.csv'
    test_path = './data/assistment_challenge/assistment_challenge_test.csv'
    save_dir_prefix = './results/assistment_challenge/'
elif dataset == 'toy':
    train_path = './data/toy_data_train.csv'
    test_path = './data/toy_data_test.csv'
    save_dir_prefix = './results/toy/'
elif dataset == 'a2009':
    train_path = './data/skill_id_train.csv'
    test_path = './data/skill_id_test.csv'
    save_dir_prefix = './results/a2009/'
elif dataset == 'assist_lsh':
    train_path = './data/assist_train_lsh_1116.csv'
    test_path = './data/assist_test_lsh_1116.csv'
    save_dir_prefix = './results/assist_lsh/'

def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    data = DKTData(train_path, test_path, batch_size=batch_size)
    data_train = data.train
    data_test = data.test
    num_problems = data.num_problems

    dkt = DKT(sess, data_train, data_test, num_problems, network_config,
              save_dir_prefix=save_dir_prefix,
              num_runs=num_runs, num_epochs=num_epochs,
              keep_prob=keep_prob, logging=True, save=True)

    # run optimization of the created model
    dkt.model.build_graph()
    dkt.run_optimization()
    # close the session
    sess.close()

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print("program run for: {0}s".format(end_time - start_time))
