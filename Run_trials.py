import numpy as np
import os
import multiprocessing as mp
import tensorflow as tf
from bgcn.BGCN_models import BGCN_model_trial

seed = 123
np.random.seed(seed)

# ================================== Experiment Settings ===============================================================

graph_generation_mode = 'Copying'
dataset = 'cora'
label_per_class_n = 5
num_partition = 2
num_processes = 4
num_processes = min(num_partition, num_processes)
trial_data_partition_seed = np.random.randint(1, 1e6, num_partition)
trial_index_all = np.arange(num_partition)

flags = tf.app.flags
FLAGS = flags.FLAGS

print("==========The hyper-parameter for this setting is=========")

# ================================== Key Bayesian-GCNN Hyperparameters =================================================

flags.DEFINE_string('dataset', dataset, 'Dataset string')  # Dataset: 'cora', 'citeseer', 'pubmed'
flags.DEFINE_integer('label_per_class_n', label_per_class_n,
                     "how many label per class we use")  #number of training labels per class
flags.DEFINE_boolean('random_data_partition', False, "use random data split of the fix data partition")  #True and False
flags.DEFINE_boolean('save_log', False, "whether to print log in the console or save log in .txt")
flags.DEFINE_string('model', 'gcn', 'Model string.')  # Graph training algorithm: 'gcn', 'gcn_cheby'
flags.DEFINE_string('graph_generation_mode', graph_generation_mode,
                    'None, Embedding or Copying')  # Whether or not activate the Bayesian graph inference block, C
flags.DEFINE_integer(
    'epochs', 300, 'Number of total training epoch'
)  # Total training epoch: 5 labels per class 400; 10 labels per class; 350; 20 labels per class: 300;
flags.DEFINE_integer('epoch_to_start_collect_weights', 240, 'starting point for weight collection'
                    )  # starting point to collect the neural network weights: (total training epoch - 60)
flags.DEFINE_integer(
    'pretrain_n', 200, 'pretain steps'
)  #To initiate the graph inference with a better starting point, 200 iterations of the vanilla GCNN is used to initiate the MMSBM model
flags.DEFINE_integer(
    'weight_sample_interval', 1,
    'the interval of the weight collection')  #decide how many weights sample we use for final prediction

# =================================== Hyperparameters for other GCN hyper-parameters ===================================

flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('learning_rate', 0.01, 'learning rate.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

# ====================================Hyperparameters for other GCN hyper-parameters====================================

if FLAGS.graph_generation_mode == 'Copying':
    flags.DEFINE_float('epsilon', 0.01, 'probability of not copying a node')
elif FLAGS.graph_generation_mode == 'Embedding':
    flags.DEFINE_float('gamma', 0.01, 'step size')
    flags.DEFINE_float('epsilon', 0.001, 'tolerance parameter')
    flags.DEFINE_integer("max_itr", 1000, "The maximum training iterations for the graph inference step")

print("============The hyper-parameters for this trial are:===========")
print("The dataset is {}".format(FLAGS.dataset))
print("The total GCN training epoch is {}".format(FLAGS.epochs))
print("The training mode is {}".format(FLAGS.graph_generation_mode))
print("The number of label for each category in the training set is {}".format(FLAGS.label_per_class_n))
print("Random data partition: {}".format(FLAGS.random_data_partition))
print("================================================================")

# ================================== Create log Directory ==============================================================

log_dir = 'log/' + FLAGS.dataset + '_' + FLAGS.model + '_' + FLAGS.graph_generation_mode + '_random_data_partition_' + str(
    FLAGS.random_data_partition) + '_' + str(FLAGS.label_per_class_n) + '_labels_' + 'training_itr_' + str(
        FLAGS.epochs) + '_' + str(FLAGS.epoch_to_start_collect_weights) + '/'

if not os.path.exists(log_dir):
    os.makedirs(os.path.dirname(log_dir))

# ================================== Start the trials ==================================================================

if __name__ == '__main__':
    pool = mp.Pool(processes=num_processes)
    pool_results = [
        pool.apply_async(BGCN_model_trial, (1, trial_data_partition_seed[indices], trial_index_all[indices], log_dir))
        for indices in range(num_partition)
    ]
    pool.close()
    pool.join()
    for pr in pool_results:
        dict_results = pr.get()
