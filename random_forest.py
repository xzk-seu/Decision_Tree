import random
import os
import json
import csv
from copy import deepcopy
from decision_tree import DecisionTree
from tqdm import tqdm
from multiprocessing import Pool


class RandomForest(object):
    def __init__(self, full_set, feature_index_set, model='id3', tree_num=1):
        self.full_set = full_set
        self.feature_index_set = feature_index_set
        self.model = model
        self.tree_num = tree_num

        if self.tree_num == 0:
            return

        test_data_id = [x['id'] for x in full_set]

        sample_size = len(full_set) // tree_num
        self.dt_list = list()
        for i in range(tree_num):
            train_set = random.choices(full_set, k=sample_size)
            train_data_id = [x['id'] for x in train_set]
            test_data_id = [x for x in test_data_id if x not in train_data_id]
            dt = DecisionTree(train_set, feature_id_list, is_random_forest=True)
            self.dt_list.append(dt)

        print('test data is: ', len(test_data_id))
        self.test_set = [x for x in full_set if x['id'] in test_data_id]

    def get_test_num(self):
        if self.tree_num == 0:
            return 0
        return len(self.test_set)

    def get_accuracy(self):
        if self.tree_num == 0:
            return 0
        correction = 0
        for data in self.test_set:
            pre_dict = dict()
            for tree in self.dt_list:
                p = tree.predict(data)
                if p not in pre_dict.keys():
                    pre_dict[p] = 0
                pre_dict[p] += 1
            prediction = max(pre_dict, key=lambda x: pre_dict[x])
            if prediction == data['class']:
                correction += 1
        accuracy = correction / len(self.test_set)
        return accuracy


def run(full_data, feature_list, i):
    try:
        # print('full_data: ', len(full_data))
        rf = RandomForest(full_data, feature_list, tree_num=i)
        a = rf.get_accuracy()
        t = rf.get_test_num()
        # csv_data.append([i, a, t])
        print([i, a, t])
        return [i, a, t]
    except Exception as e:
        print(e)


def all_run():
    pool = Pool(4)
    result = list()
    for i in tqdm(range(100, 200)):
        result.append(pool.apply_async(run, args=(FULL_DATA, feature_id_list, i)))
    pool.close()
    pool.join()

    csv_data = list()
    for r in result:
        csv_data.append(r.get())

    with open('rf_2.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(csv_data)


path = os.path.join(os.getcwd(), 'data', 'adult', 'adult_data.json')
FULL_DATA = json.load(open(path, 'r'))
sample = FULL_DATA[0]
feature_id_list = list(range(len(sample['categories_feature'])))
random.shuffle(FULL_DATA)


if __name__ == '__main__':
    all_run()
