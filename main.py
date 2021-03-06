# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 11:16:30 2020

@author: LSH
"""

import argparse
import os
import time

import tensorflow as tf

from load_data import DKTData
from utils import DKT

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()

# data file configuration
parser.add_argument("-lw1", "--lambda_w1", type=float, default=0.00,
                    help="The lambda coefficient for the regularization waviness with l1-norm.")
parser.add_argument("-lw2", "--lambda_w2", type=float, default=0.00,
                    help="The lambda coefficient for the regularization waviness with l2-norm.")
parser.add_argument("-lo", "--lambda_o", type=float, default=0.00,
                    help="The lambda coefficient for the regularization objective.")
parser.add_argument('--num_epochs', type=int, default=30,
                    help="Whether to add skill embedding layer after input.")
parser.add_argument('--dataset', type=str, default='assist2009')
parser.add_argument('--no-emb-layer', dest='emb_layer', action='store_false')
parser.add_argument('--no-recurrent-test',
                    dest='recurrent_test', action='store_false')
args = parser.parse_args()

rnn_cells = {
    "LSTM": tf.contrib.rnn.LSTMCell,
    "GRU": tf.contrib.rnn.GRUCell,
    "BasicRNN": tf.contrib.rnn.BasicRNNCell,
    "LayerNormBasicLSTM": tf.contrib.rnn.LayerNormBasicLSTMCell,
}

num_runs = 5
num_epochs = args.num_epochs
batch_size = 128
keep_prob = 0.5

network_config = {}
network_config['batch_size'] = batch_size
network_config['hidden_layer_structure'] = [200, ]
network_config['learning_rate'] = 1e-2
network_config['keep_prob'] = keep_prob
network_config['rnn_cell'] = rnn_cells['LSTM']
network_config['max_grad_norm'] = 5.0
network_config['lambda_w1'] = args.lambda_w1  # dkt plus: 0.003
network_config['lambda_w2'] = args.lambda_w2  # dkt plus: 3.0
network_config['lambda_o'] = args.lambda_o  # dkt plus: 0.1
network_config['emb_layer'] = args.emb_layer
network_config['skill_separate_emb'] = True
network_config['expand_correct_dim'] = False
network_config['embedding_dims'] = 200
network_config['recurrent_test'] = args.recurrent_test

if args.dataset == 'assist2009':
    train_path = './data/assist2009_updated/assist2009_updated_train1.csv'
    valid_path = './data/assist2009_updated/assist2009_updated_valid1.csv'
    test_path = './data/assist2009_updated/assist2009_updated_test.csv'
    save_dir_prefix = './results/assist2009_updated/'
elif args.dataset == 'assist2009_1127':
    train_path = './data/ASSISTment_skill_builder_only_1127/assistment_1127_train.csv'
    valid_path = './data/ASSISTment_skill_builder_only_1127/assistment_1127_valid.csv'
    test_path = './data/ASSISTment_skill_builder_only_1127/assistment_1127_test.csv'
    save_dir_prefix = './results/assitment_1127/'


def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    data = DKTData(train_path, valid_path, test_path, batch_size=batch_size)
    data_train = data.train
    data_valid = data.valid
    data_test = data.test
    num_problems = data.num_problems

    dkt = DKT(sess, data_train, data_valid, data_test, num_problems, network_config,
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
