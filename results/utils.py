# # -*- encoding: utf-8 -*-
# # here put the import lib
# import json
# import jsonlines
# from tqdm import tqdm
# import os


# def read_data(data_path):

#     data = []

#     with jsonlines.open(data_path, "r") as f:
#         for meta_data in f:
#             data.append(meta_data)

#     return data


# def save_data(data_path, data):
#     # write all_data list to a new jsonl
#     with jsonlines.open(data_path, "w") as w:
#         for meta_data in data:
#             w.write(meta_data)


# def extract_data(data):

#     data_dict = {}

#     for meta_data in tqdm(data):
#         if meta_data['task_dataset'] not in data_dict.keys():
#             data_dict[meta_data['task_dataset']] = []
#         data_dict[meta_data['task_dataset']].append(meta_data)
#     print("extract conpletion")

#     return data_dict


# def partition(data_dict, task_list, output_path):
#     print("begin partition")
#     for task in tqdm(task_list):
#         task_path = os.path.join(output_path, task)

#         if not os.path.exists(task_path):
#             os.makedirs(task_path)

#         if not os.path.exists(os.path.join(task_path, "test.json")):
#             with open(os.path.join(task_path, "test.json"), 'w', encoding='utf-8') as json_file:
#                 json.dump({}, os.path.join(task_path, "test.json"))  # 写入空的 JSON 对象
#         save_data(os.path.join(task_path, "test.json"), data_dict[task])
#         print("complete partition")

#正版
# -*- encoding: utf-8 -*-
# here put the import lib
import json
import jsonlines
from tqdm import tqdm
import os


def read_data(data_path):

    data = []

    with jsonlines.open(data_path, "r") as f:
        for meta_data in f:
            data.append(meta_data)

    return data


def save_data(data_path, data):
    # write all_data list to a new jsonl
    with jsonlines.open(data_path, "w") as w:
        for meta_data in data:
            w.write(meta_data)


def extract_data(data):

    data_dict = {}

    for meta_data in tqdm(data):
        if meta_data['task_dataset'] not in data_dict.keys():
            data_dict[meta_data['task_dataset']] = []
        data_dict[meta_data['task_dataset']].append(meta_data)

    return data_dict


def partition(data_dict, task_list, output_path):

    for task in task_list:
        task_path = os.path.join(output_path, task)

        if not os.path.exists(task_path):
            os.makedirs(task_path)

        save_data(os.path.join(task_path, "test_predictions.json"), data_dict[task])