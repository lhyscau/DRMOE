import json
import sys
sys.path.append("./")

def remove_spaces_from_target(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    updated_lines = []

    for line in lines:
        # 解析每一行的 JSON 数据
        data = json.loads(line)
        
        # 去除 target 键对应值中的所有空格
        if 'target' in data:
            data['target'] = data['target'].replace(" ", "")
        
        # 将修改后的 JSON 数据转换为字符串
        updated_line = json.dumps(data, ensure_ascii=False)
        
        # 添加到更新后的行列表中
        updated_lines.append(updated_line)
    
    # 将更新后的内容写回文件
    with open(file_path, 'w', encoding='utf-8') as file:
        for updated_line in updated_lines:
            file.write(updated_line + '\n')

# 调用函数处理文件
remove_spaces_from_target('./pred/test_predictions.json')
