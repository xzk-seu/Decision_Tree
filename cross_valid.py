import os
import json
import csv
from decision_tree import DecisionTree
import random
from tqdm import tqdm


def batch_process(i):
    v_start = (i + 7) % 10
    t_start = (i + 9) % 10
    train_set = list()
    valid_set = list()
    for j in range(7):
        t = (i + j) % 10
        train_set.extend(batch_list[t])
    for j in range(3):
        t = (v_start + j) % 10
        valid_set.extend(batch_list[t])
    test_set = batch_list[t_start]

    dt = DecisionTree(train_set, feature_id_list, model='C4.5')
    depth = dt.get_depth()
    leaf = dt.get_leaf_num()
    node = dt.get_node_num()
    accuracy = dt.get_accuracy(test_set)
    dt.post_pruning(valid_set)
    post_depth = dt.get_depth()
    post_leaf = dt.get_leaf_num()
    post_node = dt.get_node_num()
    post_accuracy = dt.get_accuracy(test_set)

    data = [depth, leaf, node, accuracy, post_depth, post_leaf, post_node, post_accuracy]
    return data


if __name__ == '__main__':
    path = os.path.join(os.getcwd(), 'data', 'adult', 'adult_data.json')
    full_data = json.load(open(path, 'r'))
    sample = full_data[0]
    print(sample)
    random.shuffle(full_data)
    feature_id_list = list(range(len(sample['categories_feature'])))

    batch = 10
    batch_size = len(full_data)//batch
    batch_list = list()
    for it in range(batch):
        start = it * batch_size
        if it == batch-1:
            temp = full_data[start:]
        else:
            temp = full_data[start: start + batch_size]
        batch_list.append(temp)

    result = list()
    for it in tqdm(range(batch)):
        d = batch_process(it)
        result.append(d)
    with open('cross_valid_c4_5.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(result)
