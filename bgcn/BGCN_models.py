from __future__ import division
from __future__ import print_function
from bgcn.gcn.utils_gcn import *
from bgcn.graph_inference import sample_graph_copying
from bgcn.gcn.models import GCN
import numpy as np
from scipy.sparse import csr_matrix
from datetime import datetime
import random
import sys
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
import argparse
from bgcn.utils import edges_non_edges_index
from bgcn.data_partition import data_partition_random, data_partition_fixed
import tensorflow as tf
import time
import pickle as pk
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  #the GPU number

flags = tf.app.flags
FLAGS = flags.FLAGS


# ====================================Main function for GCNN=======================================
def BGCN_model_trial(trials_per_partition, data_partition_seed, trial_index, log_dir):

    def train_gcn_one_epoch(support_graph, epoch):
        # Prepare feed dict for training set
        t = time.time()
        feed_dict_train = construct_feed_dict(features, support_graph, y_train, train_mask, placeholders)
        feed_dict_train.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict_train)

        # Validation set
        cost_val, acc_val, duration = evaluate(features, support_original, y_val, val_mask, placeholders)

        # Test set using the sampled graph
        test_cost, test_acc, test_duration = evaluate(features, support_graph, y_test, test_mask, placeholders)

        # Test set using the original graph
        test_cost_original_graph, test_acc_original_graph, _ = evaluate(features, support_original, y_test, test_mask,
                                                                        placeholders)

        #get the softmax using the sample graphs
        feed_dict_val = construct_feed_dict(features, support_graph, y_test, test_mask, placeholders)
        feed_dict_val.update({placeholders['dropout']: 0})
        soft_labels_sample_graphs = sess.run(tf.nn.softmax(model.outputs), feed_dict=feed_dict_val)

        #get the softmax of using the original graph
        feed_dict_OG = construct_feed_dict(features, support_original, y_test, test_mask, placeholders)
        feed_dict_OG.update({placeholders['dropout']: 0})
        soft_labels_OG_graph = sess.run(tf.nn.softmax(model.outputs), feed_dict=feed_dict_OG)

        # evaluate cross-entropy loss
        cross_entropy_loss_avg_soft_labels = log_loss(labels_value[test_set_index],
                                                      soft_labels_sample_graphs[test_set_index])

        #  Print results
        if epoch % 10 == 9:
            print("===================================================================")
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]), "train_acc=",
                  "{:.5f}".format(outs[2]), "time=", "{:.5f}".format(time.time() - t))

            print("val_loss=", "{:.5f}".format(cost_val), "val_acc=", "{:.5f}".format(acc_val))

        return cost_val, soft_labels_OG_graph, soft_labels_sample_graphs

    # data_partition_seed is used for data partition and generate a seed list for the neural network initial seed
    np.random.seed(data_partition_seed)
    trials_per_partition = trials_per_partition
    seed_list = np.random.randint(1, 1e6, trials_per_partition)

    for seed in seed_list:
        np.random.seed(data_partition_seed)  #decide the data partition seed
        # ===========================load data========================================
        timestamp = str(datetime.now())[0:10]
        log_file_name = 'trial_index_' + str(trial_index) + '_data_partition_seed_' + str(data_partition_seed) \
                        + '_seed_' + str(seed) + '_' + timestamp + '.txt'
        if FLAGS.save_log:
            sys.stdout = open(log_dir + log_file_name, 'w')

        if not FLAGS.random_data_partition:
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels, order = \
                data_partition_fixed(dataset_name=FLAGS.dataset, label_n_per_class=FLAGS.label_per_class_n)
        else:
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels, order = \
                data_partition_random(dataset_name=FLAGS.dataset, label_n_per_class=FLAGS.label_per_class_n)

        np.random.seed(seed)  # decide the seed for graph inference
        random.seed(seed)
        tf.set_random_seed(seed)  # decide the random seed for neural network initial weights

        print("The index number for this trial is {}".format(trial_index))
        print("The data partition seed for this trial is {}".format(data_partition_seed))
        print("The seed number for this trial is {}".format(seed))

        N = len(y_train)

        node_neighbors_dict = {}
        for i in range(N):
            node = adj[i]
            node_neighbors_dict[i] = csr_matrix.nonzero(node)[1]

        labels_value = labels.argmax(axis=1)  #get the label value

        test_set_index = np.where(test_mask == True)[0]

        # ===========================================GCNN model set up========================================
        features = preprocess_features(features)
        if FLAGS.model == 'gcn':
            support_original = [preprocess_adj(adj)]
            num_supports = 1
            model_func = GCN
        elif FLAGS.model == 'gcn_cheby':
            support_original = chebyshev_polynomials(adj, FLAGS.max_degree)
            num_supports = 1 + FLAGS.max_degree
            model_func = GCN
        else:
            raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

        # Define placeholders
        placeholders = {
            'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
            'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
            'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
            'labels_mask': tf.placeholder(tf.int32),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'num_features_nonzero':
                tf.placeholder(tf.int32)  # helper variable for sparse dropout
        }

        order = labels_value.argsort()

        model = model_func(placeholders, input_dim=features[2][1], logging=True)
        upper_tri_index = np.triu_indices(N, k=1)

        # =======================================GCNN model initialization==============================================
        # Initialize session

        sess = tf.Session()

        # Define model evaluation function
        def evaluate(features, support, labels, mask, placeholders):
            t_test = time.time()
            feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
            outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
            return outs_val[0], outs_val[1], (time.time() - t_test)

        # Init variables
        sess.run(tf.global_variables_initializer())

        cost_val_list = []

        soft_max_sum_OG_graph = 0  # store the softmax output of the GCNN from different weight samples (from original graph)
        soft_max_sum_sample_graph = 0  # store the softmax output of the GCNN from different weight samples (from sample graphs)

        print("===============================Start training the GCNN Model========================")
        for epoch in range(FLAGS.epochs):
            if FLAGS.graph_generation_mode == 'Copying':
                # =======================================GCNN pre train process=====================================
                if epoch < FLAGS.pretrain_n:
                    cost_val, soft_labels_OG_graph, soft_labels_sample_graphs = train_gcn_one_epoch(
                        support_original, epoch)
                    cost_val_list.append(cost_val)

                    obtained_labels = soft_labels_OG_graph.argmax(axis=1)

                if epoch == FLAGS.pretrain_n:

                    sampled_graph = sample_graph_copying(
                        seed, node_neighbors_dict, obtained_labels, order, set_seed=True)

                    inferred_graph = csr_matrix(sampled_graph)
                    # pk.dump(MAP_graph, open(os.path.join(log_dir, MAP_graph), 'wb'))

                    if FLAGS.model == 'gcn_cheby':
                        support = chebyshev_polynomials(inferred_graph, FLAGS.max_degree)
                    else:
                        support = [preprocess_adj(inferred_graph)]

                    cost_val, soft_labels_OG_graph, soft_labels_sample_graphs = train_gcn_one_epoch(support, epoch)
                    cost_val_list.append(cost_val)

                if epoch > FLAGS.pretrain_n:

                    sampled_graph = sample_graph_copying(
                        seed, node_neighbors_dict, obtained_labels, order, set_seed=False)

                    inferred_graph = csr_matrix(sampled_graph)
                    # pk.dump(MAP_graph, open(os.path.join(log_dir, MAP_graph), 'wb'))

                    if FLAGS.model == 'gcn_cheby':
                        support = chebyshev_polynomials(inferred_graph, FLAGS.max_degree)
                    else:
                        support = [preprocess_adj(inferred_graph)]

                    cost_val, soft_labels_OG_graph, soft_labels_sample_graphs = train_gcn_one_epoch(support, epoch)

                    # ===========save the softmax output from different weight samples=================
                    if epoch > FLAGS.epoch_to_start_collect_weights and epoch % FLAGS.weight_sample_interval == 0:

                        soft_max_sum_OG_graph += soft_labels_OG_graph
                        hard_label_OG_graph = soft_max_sum_OG_graph.argmax(axis=1)
                        acc_OG_graph = accuracy_score(labels_value[test_set_index], hard_label_OG_graph[test_set_index])

                        soft_max_sum_sample_graph += soft_labels_sample_graphs
                        hard_label_sample_graph = soft_max_sum_sample_graph.argmax(axis=1)
                        acc_sample_graph = accuracy_score(labels_value[test_set_index],
                                                          hard_label_sample_graph[test_set_index])

                        obtained_labels = hard_label_sample_graph

                        if epoch % 10 == 9:
                            print("============= weight sampling results at iteration {}==========".format(epoch + 1))
                            print("The accuracy from avg weight sampling using the original graph is {}".format(
                                acc_OG_graph))
                            print("The accuracy from avg weight sampling using the sample graph is {}".format(
                                acc_sample_graph))

                    cost_val_list.append(cost_val)
            elif FLAGS.graph_generation_mode == 'None':
                cost_val, soft_labels_OG_graph, soft_labels_sample_graphs = train_gcn_one_epoch(support_original, epoch)
                cost_val_list.append(cost_val)

                if epoch > FLAGS.epoch_to_start_collect_weights and epoch % FLAGS.weight_sample_interval == 0:

                    soft_max_sum_OG_graph += soft_labels_OG_graph
                    hard_label_OG_graph = soft_max_sum_OG_graph.argmax(axis=1)
                    acc_OG_graph = accuracy_score(labels_value[test_set_index], hard_label_OG_graph[test_set_index])

                    if epoch % 10 == 9:
                        print("========= weight sampling results at iteration {}========".format(epoch + 1))
                        print(
                            "The accuracy from avg weight sampling using the original graph is {}".format(acc_OG_graph))

            else:
                raise ValueError('Invalid argument for model: ' + str(FLAGS.graph_generation_mode))

        print("Optimization Finished!")
        print("===============================Start evaluate the final model performance =============================")

        if FLAGS.graph_generation_mode != 'None':
            softmax_log_file_name = 'BCGN_softmax_trial_index_' + str(trial_index) + '_data_partition_seed_' + str(
                data_partition_seed) + '_seed_' + str(seed) + '_' + timestamp + '.pk'
            pk.dump(soft_max_sum_OG_graph, open(os.path.join(log_dir, softmax_log_file_name), 'wb'))
            # Test set using the sampled graph
            test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)

            # Test set using the original graph
            test_cost_original_graph, test_acc_original_graph, _ = evaluate(features, support_original, y_test,
                                                                            test_mask, placeholders)
            print("Model: Bayesian GCNN")
            print("============1) final result using the weight at the last iteration (OG graph)=========")
            print("The accuracy from original graph is {}".format(test_acc_original_graph))

            print("============1) final result using the weight at the last iteration (Sample graph)=========")
            print("The accuracy from original graph is {}".format(test_acc))

            print("============2) final result for weight samping and graph sampling=========")
            print("The accuracy from avg weight sampling using the original graph is {}".format(acc_OG_graph))
            print("The accuracy from avg weight sampling using the sample graph is {}".format(acc_sample_graph))

        else:
            softmax_log_file_name = 'Vanilla_softmax_trial_index_' + str(trial_index) + '_data_partition_seed_' + str(
                data_partition_seed) + '_seed_' + str(seed) + '_' + timestamp + '.pk'
            pk.dump(soft_max_sum_OG_graph, open(os.path.join(log_dir, softmax_log_file_name), 'wb'))
            test_cost_original_graph, test_acc_original_graph, _ = evaluate(features, support_original, y_test,
                                                                            test_mask, placeholders)
            print("Model: Vanilla GCNN")
            print("============1) final result using the weight at the last iteration=========")
            print("The accuracy from original graph is {}".format(test_acc_original_graph))
            print("============2) final result for weight samping===========")
            print("The accuracy from avg weight sampling using the original graph is {}".format(acc_OG_graph))

        sess.close()
        tf.reset_default_graph()
    return 0
