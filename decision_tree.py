import json
import os
from math import log2
from copy import deepcopy


class DecisionTree(object):
    def __init__(self, train_set, feature_index_set, model='id3', with_post_purning=False):
        """
        根据训练集构建决策树
        :param train_set:全量训练数据集
        :param feature_index_set: 所选取的特征的集合
        """
        print('construct tree, index:')
        train_index_set = [x['id'] for x in train_set]
        print('len: ', len(train_index_set))
        self.train_set = train_set
        self.feature_index_set = feature_index_set
        self.model = model
        self.with_post_purning = with_post_purning
        self.count_class = partition_by_class(self.train_set)
        self.majority = max(self.count_class, key=lambda x: len(self.count_class[x]))
        self.is_leaf = False
        self.best_feature = None
        self.prediction = None
        self.sub_trees = None

        if not self.is_need_to_partition():
            # 如果不需要划分，则对该节点作为一个叶子节点处理
            self.is_leaf = True
            # print('不需要划分')
            print('预测值为:', self.prediction)
            # print('===========================')
            return

        self.best_feature = self.get_best_feature(self.model)
        print('get_best_feature: ', self.best_feature)
        self.count_best_feature = partition_by_feature(self.train_set, self.best_feature)

        if len(self.count_best_feature) == 1:
            # 按当前特征无法划分数据集
            self.is_leaf = True
            # value = self.count_best_feature.popitem()[0]
            # print('按当前特征无法划分')
            # print('因为当前数据在特征[%d]下取值都为%s' % (self.best_feature, value))
            self.prediction = self.majority
            print('预测值为:', self.prediction)
            # print('===================================')
            return

        self.sub_trees = list()
        self.branch()

    def branch(self):
        for feature_value, index_list in self.count_best_feature.items():
            if feature_value == '?':
                # 对于空值，不再建立分支
                continue
            # print('特征:%d，取值: %s' % (self.best_feature, feature_value))
            new_feature_set = deepcopy(self.feature_index_set)
            new_feature_set.remove(self.best_feature)
            new_train_set = [x for x in self.train_set if x['id'] in index_list]
            tree = DecisionTree(new_train_set, new_feature_set, self.model)
            temp_dict = dict(feature=self.best_feature, value=feature_value, tree=tree)
            self.sub_trees.append(temp_dict)

    def get_best_feature(self, model='ID3'):
        tuple_list = list()
        for feature in self.feature_index_set:
            criteria = get_criteria(model, self.train_set, feature)
            temp_tuple = (feature, criteria)
            print('feature: %d, criteria: %f' % (feature, criteria))
            tuple_list.append(temp_tuple)
        best_feature = max(tuple_list, key=lambda x: x[1])[0]
        max_criteria = -1
        for feature, criteria in tuple_list:
            if criteria >= max_criteria:
                max_criteria = criteria
                best_feature = feature
        # print('当前数据下的最优特征为: %d' % best_feature)
        return best_feature

    def is_need_to_partition(self):
        """
        判断当前节点是否需要继续划分
        :return:
        """
        if len(self.train_set) == 0:
            # 当前节点对应数据集为空，不需要划分
            self.prediction = None
            # print('当前节点对应数据集为空，不需要划分')
            return False
        if len(self.count_class) == 1:
            # 当前数据集包含的样本属于同一类，不需要划分
            self.prediction = self.count_class.popitem()[0]
            # print('当前数据集包含的样本属于同一类，不需要划分')
            return False
        current_feature_list = self.train_set[0]['categories_feature']
        for data in self.train_set[1:]:
            # 当前数据特征不完全一样，还需要继续划分
            if data['categories_feature'] != current_feature_list:
                return True
            else:
                current_feature_list = data['categories_feature']
        # 当前数据特征完全一样，无法继续划分
        # print('当前数据特征完全一样，无法继续划分')
        self.prediction = self.majority
        return False

    def get_leaf_num(self):
        if self.is_leaf:
            return 1
        else:
            leaf_num = 0
            for tree_dict in self.sub_trees:
                tree = tree_dict['tree']
                leaf_num += tree.get_leaf_num()
            return leaf_num

    def get_node_num(self):
        if self.is_leaf:
            return 1
        else:
            node_num = 1
            for tree_dict in self.sub_trees:
                tree = tree_dict['tree']
                node_num += tree.get_node_num()
            return node_num

    def get_depth(self):
        if self.is_leaf:
            return 1
        else:
            max_sub_depth = -1
            for tree_dict in self.sub_trees:
                tree = tree_dict['tree']
                temp_depth = tree.get_depth()
                if temp_depth > max_sub_depth:
                    max_sub_depth = temp_depth
            depth = 1 + max_sub_depth
            return depth

    def get_tree_dict(self):
        temp_dict = dict()
        if self.is_leaf:
            temp_dict['prediction'] = self.prediction
            return temp_dict
        if not self.sub_trees:
            return temp_dict
        temp_dict['best_feature'] = self.best_feature
        temp_dict['sub_trees'] = list()
        for tree in self.sub_trees:
            data = tree['tree'].get_tree_dict()
            temp_dict['sub_trees'].append(dict(value=tree['value'], tree=data))
        return temp_dict

    def predict(self, data):
        """
        在当前树下预测data属于哪一类
        :param data:
        :return:
        """
        if self.is_leaf:
            # print('data[%d]的预测值为:%s' % (data['id'], self.prediction))
            return self.prediction
        else:
            # print('此时判断特征【%d】' % self.best_feature)
            value = data['categories_feature'][self.best_feature]
            # print('data[%d]的feature[%d]为%s' % (data['id'], self.best_feature, value))
            for tree in self.sub_trees:
                if tree['value'] == value:
                    return tree['tree'].predict(data)

    def get_accuracy(self, test_set):
        """
        批量进行预测，返回准确率accuracy
        :param test_set:
        :return:
        """
        correction = 0
        for data in test_set:
            if data['class'] == self.predict(data):
                correction += 1
        accuracy = correction / len(test_set)
        return accuracy

    def get_sub_trees(self):
        sub_trees = list()
        if self.sub_trees:
            for t in self.sub_trees:
                sub_trees.append(t)
        return sub_trees

    def pruning(self, valid_set):
        """
        把当前节点的叶节点删除
        :return:
        """
        correct = [x for x in valid_set if x['class'] == self.majority]
        pre_accuracy = len(correct) / len(valid_set)
        post_accuracy = self.get_accuracy(valid_set)
        # print('post_accuracy: ', post_accuracy)
        # print('pre_accuracy: ', pre_accuracy)
        if post_accuracy < pre_accuracy:
            # print('执行剪枝！')
            self.prediction = self.majority
            self.is_leaf = True
            self.sub_trees = None

    def post_pruning(self, valid_set):
        if self.is_leaf:
            print('当前为叶节点，无需剪枝。')
            return
        if self.get_depth() > 2:
            for tree_dict in self.sub_trees:
                tree = tree_dict['tree']
                if tree.get_depth() < 2:
                    continue
                else:
                    tree.post_pruning(valid_set)
        elif self.get_depth() == 2:
            self.pruning(valid_set)
            if self.is_leaf:
                return
        self.pruning(valid_set)


def get_criteria(model, data_set, feature):
    if model == 'C4.5' or model == 'c4.5':
        return get_information_gain_ratio(data_set, feature)
    else:
        return get_information_gain(data_set, feature)


def partition_by_feature(data_set, feature):
    """
        按特征对数据进行划分
        :param data_set: 按类划分data_set数据集
        :param feature: 按类划分data_set数据集
        :return: 一个字典，键为一个类名，值为该类对应的数据id
        """
    r_dict = dict()
    for data in data_set:
        f = data['categories_feature'][feature]
        if f not in r_dict.keys():
            r_dict[f] = list()
        r_dict[f].append(data['id'])
    return r_dict


def partition_by_class(data_set):
    """
    按类对数据进行划分
    :param data_set: 按类划分data_set数据集
    :return: 一个字典，键为一个类名，值为该类对应的数据id
    """
    r_dict = dict()
    for data in data_set:
        c = data['class']
        if c not in r_dict.keys():
            r_dict[c] = list()
        r_dict[c].append(data['id'])
    return r_dict


def get_entropy(data_set):
    """
    计算data_set数据集的熵
    :param: 待计算的数据集
    :return:熵
    """
    count_class = partition_by_class(data_set)
    result = 0
    for c, c_list in count_class.items():
        freq = len(c_list) / len(data_set)
        result -= freq * log2(freq)
    return result


def get_feature_entropy(data_set, feature):
    old_data_set = data_set
    data_set = get_clean_data(data_set, feature)
    clean_ratio = len(data_set) / len(old_data_set)
    count_feature = partition_by_feature(data_set, feature)
    result = 0
    for k, v in count_feature.items():
        freq = len(v) / len(data_set)
        result -= freq * log2(freq)
    return result * clean_ratio


def get_condition_entropy(data_set, feature):
    """
    计算当前数据集在给定特征feature下的条件熵
    :param data_set: 待计算的数据集
    :param feature: 给定的特征
    :return: 条件熵
    """
    result = 0
    count_feature = partition_by_feature(data_set, feature)
    for feature_value, index_list in count_feature.items():
        # feature_value为特征的一个取值
        # index_list为该取值对应的样本id
        current_data = [x for x in data_set if x['id'] in index_list]
        freq = len(current_data) / len(data_set)
        temp_entropy = get_entropy(current_data)
        result += freq * temp_entropy
    return result


def get_information_gain(data_set, feature):
    """
    计算当前数据集在给定特征feature下的条件熵，考虑缺失值
    :param data_set: 待计算的数据集
    :param feature: 给定的特征
    :return: 信息增益
    """
    old_data_set = data_set
    data_set = get_clean_data(data_set, feature)
    clean_ratio = len(data_set) / len(old_data_set)
    # print('clean_ratio: ', clean_ratio)
    entropy = get_entropy(data_set)
    condition_entropy = get_condition_entropy(data_set, feature)
    information_gain = entropy - condition_entropy
    information_gain *= clean_ratio
    return information_gain


def get_information_gain_ratio(data_set, feature):
    information_gain = get_information_gain(data_set, feature)
    intrinsic_value = get_feature_entropy(data_set, feature)
    if intrinsic_value == 0:
        return 0
    return information_gain/intrinsic_value


def get_clean_data(data_set, feature):
    """
    过滤data_set在给定feature上存在缺失值的数据
    :param data_set:
    :param feature:
    :return: 不存在缺失值的数据
    """
    result = [x for x in data_set if x['categories_feature'][feature] != '?']
    return result


if __name__ == '__main__':
    print('hello')
