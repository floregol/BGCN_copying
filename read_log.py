import glob
import os
import re
import numpy as np
# import matplotlib.pyplot as plt
import re
# from matplotlib import rcParams

data_record = []
i = 0

dir = '/home/flo/git/BGCN_embed/log/pubmed_gcn_Embedding_random_data_partition_False_5_labels_training_itr_300_240/'
os.chdir(dir)
file_list = []
for file in glob.glob("*.txt"):
    file_list.append(file)
test_acc = []

test_acc_matchers = ['The accuracy from avg weight sampling using the original graph is ']

i = 0
for filename in file_list:
    with open(dir + filename, 'r') as f:
        data = f.readlines()
    print(filename)
    print(i)
    try:
        record_test_acc = [s for s in data if any(xs in s for xs in test_acc_matchers)][-1]
        numbers_loss_acc_last = re.findall(r"[-+]?\d*\.\d+|\d+", record_test_acc)

        acc = float(numbers_loss_acc_last[-1])
        print(acc)

        test_acc.append(acc)
    except Exception as e:
        pass
    i+=1


mean_acc = np.mean(test_acc)
std_acc = np.std(test_acc)

print("The accuracy and std of the from the last GAT model during training is {} and {}".format(mean_acc*100, std_acc*100))

