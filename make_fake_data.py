# import json

# data = {
#     "input": "美国1997年的报道显示早产儿的发生率为7. 5%，国内一般报道早产儿的发生率为5%～8%。\n问题：句子中的辅助检查，辅助治疗，发病率等关系类型三元组是什么？\n答：",
#     "target": "具有辅助检查关系的头尾实体对如下：\n具有辅助治疗关系的头尾实体对如下：\n具有发病率关系的头尾实体对如下：头实体为早产，尾实体为5%～8%。",
#     "answer_choices": ["辅助检查", "辅助治疗", "发病率"],
#     "task_type": "spo_generation",
#     "task_dataset": "CMeIE",
#     "sample_id": "train-87041"
# }

# # 复制2000次
# data_list = [data] * 20000

# # 将数据写入train.json文件
# with open('./train.json', 'w', encoding='utf-8') as f:
#     for item in data_list:
#         json.dump(item, f, ensure_ascii=False)
#         f.write('\n')



import pickle

# 指定文件路径
file_path = '/root/lhy/MOELoRA-peft/datasets/processed_train_dataset.pkl'

# 加载 pkl 文件
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# 检查行数
num_lines = len(data)
print(f"The number of lines in the pkl file is: {num_lines}")
print(data)
