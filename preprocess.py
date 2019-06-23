import pandas as pd
import os
import json


def data_process():
    feature_path = os.path.join(os.getcwd(), 'data', 'adult', 'feature.json')
    feature_data = json.load(open(feature_path, 'r'))
    feature = list(feature_data.keys())
    print(len(feature), feature)
    continuous_list = list()
    for n, f in enumerate(feature_data):
        if feature_data[f]['is_continuous']:
            continuous_list.append(n)
    path = os.path.join(os.getcwd(), 'data', 'adult', 'adult.data')
    df = pd.read_csv(path, header=None)
    result_list = list()
    for index, row in df.iterrows():
        temp_dict = dict()
        temp_dict['id'] = index
        row = row.tolist()
        temp_dict['class'] = row.pop(-1).strip(' ')
        temp_dict['continuous_feature'] = list()
        temp_dict['categories_feature'] = list()
        for n, item in enumerate(row):
            if n in continuous_list:
                temp_dict['continuous_feature'].append(item)
            else:
                temp_dict['categories_feature'].append(item.strip(' '))
        result_list.append(temp_dict)
    out_path = os.path.join(os.getcwd(), 'data', 'adult', 'adult_data.json')
    json.dump(result_list, open(out_path, 'w'))


def schema_process():
    feature_dict = dict()
    path = os.path.join(os.getcwd(), 'data', 'adult', 'schema')
    with open(path, 'r') as fr:
        for line in fr.readlines():
            temp = line.strip('.\n').split(': ')
            feature = temp[0]
            value = temp[1].split(', ')
            value_dict = dict()
            if 'continuous' in value:
                value_dict['is_continuous'] = True
                value_dict['value'] = None
            else:
                value_dict['is_continuous'] = False
                value_dict['value'] = value
            feature_dict[feature] = value_dict
    out_path = os.path.join(os.getcwd(), 'data', 'adult', 'feature.json')
    json.dump(feature_dict, open(out_path, 'w'))


if __name__ == '__main__':
    data_process()
