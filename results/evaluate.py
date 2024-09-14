import jsonlines
import os
import sys
sys.path.append(".")
import numpy as np
import pandas as pd
# from utils import read_data, extract_data, partition
from results.evaluation import calculate_score, process_CTC


def func(pp: str = "/root/lhy/MOELoRA-peft/results/pred/test_predictions.json", 
         tp: str = "/root/lhy/MOELoRA-peft/results/true/test.json"):   #关键字参数
    
    target_list = ["test_predictions.json"]
    task_list=('CMeIE', 'CHIP-CDN', 'CHIP-CDEE', 'CHIP-MDCFNPC',
           'CHIP-CTC', 'KUAKE-QIC',
           'IMCS-V2-MRG', 'MedDG',)
    score_dict = {target: [] for target in target_list}
    label_dict = {}

    for target in target_list:
        for task in task_list:
            if task == "CHIP-CTC":  # CTC needs post process
                post_process_function = process_CTC
            else:
                post_process_function = None

            score, labels, _ = calculate_score(task, pp, tp, post_process_function)
            score_dict[target].append(score)
            label_dict[task] = labels

    res_data, res_key = [], []
    for key, value in score_dict.items():
        res_data.append(value)
        res_key.append(key)

    res_df = pd.DataFrame(columns=task_list,
                        index=res_key,
                        data=res_data)
    res_df["average"] = res_df.mean(axis=1)

    # 去除索引列 "test_predictions.json"
    res_df.reset_index(drop=True, inplace=True)

    eval_metric_file = "results/eval_metric.csv"
    print(res_df.head(20))
    if not os.path.exists(eval_metric_file):
        res_df.to_csv(eval_metric_file, mode='w', index=False)
    else:
        res_df.to_csv(eval_metric_file, mode='a', header=False, index=False)

    res_dict = res_df.to_dict()
    return res_dict


if __name__ == "__main__":
    import sys
    pp = sys.argv[1]
    # output_path = sys.argv[2]
    func(pp=pp)


#Prediction
# task_list=('CMeIE', 'CHIP-CDN', 'CHIP-CDEE', 'CHIP-MDCFNPC',
#            'CHIP-CTC', 'KUAKE-QIC',
#            'IMCS-V2-MRG', 'MedDG',)

# target_list = ["test_predictions.json"]

# score_dict = {target: [] for target in target_list}

# label_dict = {}

# for target in target_list:
#     for task in task_list:
#         pp = "/root/lhy/MOELoRA-peft/results/pred/tiny_test_predictions.json"
#         tp = "/root/lhy/MOELoRA-peft/datasets/tiny_test.json"
#         if task == "CHIP-CTC":  # CTC needs post process
#             post_process_function = process_CTC
#         else:
#             post_process_function = None

#         score, labels, _ = calculate_score(task, pp, tp, post_process_function)
#         score_dict[target].append(score)
#         label_dict[task] = labels


# res_data, res_key = [], []
# for key, value in score_dict.items():
#     res_data.append(value)
#     res_key.append(key)

# res_df = pd.DataFrame(columns=task_list,
#                       index=res_key,
#                       data=res_data)
# res_df["average"] = res_df.mean(axis=1)

# print(res_df.head(20))



# import jsonlines
# import os
# import sys
# sys.path.append(".")
# import numpy as np
# import pandas as pd
# from utils import read_data, extract_data, partition
# from evaluation import calculate_score, process_CTC


# task_list=('CMeIE', 'CHIP-CDN', 'CHIP-CDEE', 'CHIP-MDCFNPC',
#            'CHIP-CTC', 'KUAKE-QIC',
#            'IMCS-V2-MRG', 'MedDG',)
# pred_path = "/mnt/e/Project/MOELoRA-peft/results/pred"
# true_path = "/mnt/e/Project/MOELoRA-peft/results/true"


# target_list = ["test_predictions.json"]

# score_dict = {target: [] for target in target_list}

# score_dict = {target: [] for target in target_list}
# label_dict = {}

# for target in target_list:
#     # target_path = os.path.join(pred_path, target)   #pred/task_predictions.json
#     target_path = "/root/lhy/MOELoRA-peft/results/pred/test_predictions.json"
    
#     if not os.path.exists(os.path.join(pred_path, task_list[0])):  # needs partition
#         # all_data = read_data(os.path.join(os.path.join(pred_path, "test_predictions.json"))
#         all_data = read_data(target_path)
#         # all_data = read_data(os.path.join("./", target_path))
#         partition(extract_data(all_data), task_list, pred_path)
    
#     for task in task_list:

#         # pp = os.path.join(target_path, task)
#         # tp = os.path.join(true_path, task)
#         # pp = os.path.join("pred", "test_predictions.json")      #pred_path
#         # tp = os.path.join("true", "test.json")       #truth_path
# # /mnt/e/Project/MOELoRA-peft/results/pred/test_predictions.json
#         pp = "/root/lhy/MOELoRA-peft/results/pred/ckpt-5000/test_predictions.json"
#         tp = "/root/lhy/MOELoRA-peft/results/true/test.json"
#         if task == "CHIP-CTC":  # CTC needs post process
#             post_process_function = process_CTC
#         else:
#             post_process_function = None

#         score, labels, _ = calculate_score(task, pp, tp, post_process_function)
#         score_dict[target].append(score)
#         label_dict[task] = labels


# # import tqdm
# # # 假设 read_data, save_data, extract_data, partition 函数已经定义
# # # 请确保这些函数的实现与您之前提供的代码一致

# # # 假设 pred_path 和 task_list 已经被定义
# # # pred_path = '/path/to/predictions'  # 预测结果的根目录路径
# # # task_list = ['task1', 'task2', 'task3']  # 任务列表

# # # 定义 read_data 函数（示例）
# # def read_data(data_path):
# #     with jsonlines.open(data_path, "r") as f:
# #         data = [meta_data for meta_data in f]
# #     return data

# # # 定义 save_data 函数（示例）
# # def save_data(data_path, data):
# #     with jsonlines.open(data_path, "w") as w:
# #         for meta_data in data:
# #             w.write(meta_data)

# # # 定义 extract_data 函数（示例）
# # def extract_data(data):

# #     data_dict = {}

# #     for meta_data in data:
# #         if meta_data['task_dataset'] not in data_dict.keys():
# #             data_dict[meta_data['task_dataset']] = []
# #         data_dict[meta_data['task_dataset']].append(meta_data)
# #     print("extract conpletion")

# #     return data_dict

# # # 定义 partition 函数（示例）
# # def partition(data_dict, task_list, output_path):
# #     for task in task_list:
# #         task_path = os.path.join(output_path, task)
# #         if not os.path.exists(task_path):
# #             os.makedirs(task_path)

# #         save_data(os.path.join(task_path, 'test.json'), data_dict[task])
# #         # save_data(os.path.join(task_path, "lhy/MOELoRA-peft/datasets/test.json"), data_dict[task])

# # # 主脚本逻辑
# # # 读取预测结果
# # target_path = "/root/lhy/MOELoRA-peft/results/true/test.json"  # 预测结果文件路径
# # all_data = read_data(target_path)

# # # 提取和组织数据
# # extracted_data = extract_data(all_data)

# # # 分区并保存数据
# # partition(extracted_data, task_list, true_path)



# res_data, res_key = [], []
# for key, value in score_dict.items():
#     res_data.append(value)
#     res_key.append(key)

# res_df = pd.DataFrame(columns=task_list,
#                       index=res_key,
#                       data=res_data)
# res_df["average"] = res_df.mean(axis=1)


# print(res_df.head(20))

# # try:
# #     new_res_df = res_df.drop(columns=["CHIP-STS", "KUAKE-IR", "average"])
# # except:
# #     new_res_df = res_df
# # new_res_df["average"] = new_res_df.mean(axis=1)
# # new_res_df.head(50)