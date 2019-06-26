import os
import json
from decision_tree import DecisionTree
import random


def temp_function():
    batch = list()
    for i in range(10):
        path = os.path.join(os.getcwd(), 'data', 'data_%d.json' % i)
        batch.append(json.load(open(path, 'r')))
    feature_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

    a_list = list()
    for train_set in batch:
        dt = DecisionTree(train_set, feature_list)
        t = dt.get_tree_dict()
        test_set = batch[-1]
        correction = 0
        for data in test_set:
            if data['class'] == dt.predict(data):
                correction += 1
        accuracy = correction / len(test_set)
        print('accuracy', accuracy)
        a_list.append(accuracy)
    print(a_list)

    # json.dump(t, open('p.json', 'w'))
    #
    # path = os.path.join(os.getcwd(), 'data', 'data_%d.json' % 9)
    # test_set = json.load(open(path, 'r'))
    # correction = 0
    # for data in test_set:
    #     if data['class'] == dt.predict(data):
    #         correction += 1
    # accuracy = correction/len(test_set)
    # print('accuracy', accuracy)


def batch_test():
    path = os.path.join(os.getcwd(), 'data', 'adult', 'adult_data.json')
    full_data = json.load(open(path, 'r'))
    sample = full_data[0]
    print(sample)
    random.shuffle(full_data)
    train_set = full_data[: 1000]
    test_set = full_data[-1000:]
    valid_set = full_data[300: 400]
    dt = DecisionTree(train_set, list(range(len(sample['categories_feature']))))
    cdt = DecisionTree(train_set, list(range(len(sample['categories_feature']))), model='c4.5')
    t = dt.get_tree_dict()
    json.dump(t, open('test.json', 'w'))
    a = dt.get_accuracy(test_set)
    ca = cdt.get_accuracy(test_set)
    print('accuracy %.10f' % a)
    print('depth, ', dt.get_depth())
    print('leaf num, ', dt.get_leaf_num())
    print('node num, ', dt.get_node_num())
    print('======================')
    print('accuracy %.10f' % ca)
    print('depth, ', cdt.get_depth())
    print('leaf num, ', cdt.get_leaf_num())
    print('node num, ', cdt.get_node_num())
    print('======================')
    dt.post_pruning(valid_set)
    a = dt.get_accuracy(test_set)
    print('accuracy %.10f' % a)
    print('depth, ', dt.get_depth())
    print('leaf num, ', dt.get_leaf_num())
    print('node num, ', dt.get_node_num())
    cdt.post_pruning(valid_set)
    print('======================')
    print('accuracy %.10f' % cdt.get_accuracy(test_set))
    print('depth, ', cdt.get_depth())
    print('leaf num, ', cdt.get_leaf_num())
    print('node num, ', cdt.get_node_num())


if __name__ == '__main__':
    path = os.path.join(os.getcwd(), 'data', 'adult', 'adult_data.json')
    full_data = json.load(open(path, 'r'))
    sample = full_data[0]
    print(sample)
    random.shuffle(full_data)
    print(len(full_data))


