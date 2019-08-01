import numpy as np
from bgcn.gcn.utils_gcn import load_data


def data_partition_random(dataset_name, label_n_per_class):
    text_set_n = 1000
    val_set_n = 500

    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, one_hot_labels, order = load_data(dataset_name)

    N = len(y_train) #the sample number
    K = len(y_train[0]) # the class number

    labels = one_hot_labels.argmax(axis=1)

    train_index_new = np.zeros(K*label_n_per_class).astype(int)
    train_mask_new = np.zeros(N).astype(bool)
    val_mask_new = np.zeros(N).astype(bool)
    test_mask_new = np.zeros(N).astype(bool)

    y_train_new = np.zeros((N, K))
    y_val_new = np.zeros((N, K))
    y_test_new = np.zeros((N, K))

    class_index_dict = {}
    for i in range(K):
        class_index_dict[i] = np.where(labels == i)[0]

    for i in range(K):
        class_index = class_index_dict[i]
        train_index_one_class = np.random.choice(class_index, label_n_per_class, replace=False)
        print("The training set index for class {} is {}".format(i, train_index_one_class))
        train_index_new[i*label_n_per_class:i*label_n_per_class + label_n_per_class] = train_index_one_class

    train_index_new = list(train_index_new)
    test_val_potential_index = list(set([i for i in range(N)]) - set(train_index_new))
    test_index_new = np.random.choice(test_val_potential_index, text_set_n, replace=False)
    potential_val_index = list(set(test_val_potential_index) - set(test_index_new))
    val_index_new = np.random.choice(potential_val_index, val_set_n, replace=False)

    train_mask_new[train_index_new] = True
    val_mask_new[val_index_new] = True
    test_mask_new[test_index_new] = True

    for i in train_index_new:
        y_train_new[i][labels[i]] = 1

    for i in val_index_new:
        y_val_new[i][labels[i]] = 1

    for i in test_index_new:
        y_test_new[i][labels[i]] = 1

    return adj, features, y_train_new, y_val_new, y_test_new, train_mask_new, val_mask_new, test_mask_new, one_hot_labels, order


def data_partition_fixed(dataset_name, label_n_per_class):
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, one_hot_labels, order = load_data(
        dataset_name)
    K = len(y_train[0])

    train_set_index = np.where(train_mask == True)[0]

    labels = one_hot_labels.argmax(axis=1)

    train_set_labels = labels[train_set_index]
    train_node_index = {}
    for i in range(K):
        train_node_index[i] = np.where(train_set_labels == i)[0]

    for i in range(K):
        hide_index = train_node_index[i][label_n_per_class:]
        print("The training set index for class {} is {}".format(i, train_node_index[i][0:label_n_per_class]))
        train_mask[hide_index] = False
        y_train[hide_index] = 0

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, one_hot_labels, order
