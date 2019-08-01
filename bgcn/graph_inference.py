import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from scipy.sparse import coo_matrix
import random
import itertools
from scipy.sparse import coo_matrix

import sys

import tensorflow as tf
import time
import pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS


def sample_graph_copying(seed, node_neighbors_dict, labels, order, set_seed=False):

    if set_seed:
        np.random.seed(seed)  # decide the seed for graph inference
        random.seed(seed)

    N = len(labels)
    K = np.max(labels) + 1

    labels_dict = {}
    for k in range(K):
        labels_dict[k] = np.where(labels == k)[0]

    for i in range(N):

        if random.uniform(0, 1) < 1-FLAGS.epsilon:

            sampled_node = np.random.choice(labels_dict[labels[i]], 1)[0]

        else:

            sampled_node = i

        row_index_i = node_neighbors_dict[sampled_node]
        col_index_i = i * np.ones(len(row_index_i))
        col_index_i = col_index_i.astype(int)

        if i == 0:
            row_index = row_index_i
            col_index = col_index_i
        else:
            row_index = np.concatenate((row_index, row_index_i), axis=0)
            col_index = np.concatenate((col_index, col_index_i), axis=0)

    link_index_row = np.concatenate((row_index, col_index), axis=0)
    link_index_col = np.concatenate((col_index, row_index), axis=0)
    data = np.ones(len(link_index_row))

    sampled_graph = coo_matrix((data, (link_index_row, link_index_col)), shape=(N, N))

    # sampled_graph_dense = np.zeros((N, N))
    # sampled_graph_dense[link_index_row, link_index_col] = 1
    # plt.imshow(sampled_graph_dense[np.ix_(order, order)])
    # plt.colorbar()
    # plt.show()
    # plt.show(block=False)
    # time.sleep(5)
    # plt.close('all')

    return sampled_graph
