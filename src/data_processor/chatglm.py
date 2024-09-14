# -*- encoding: utf-8 -*-
# here put the import lib
import json
import torch
from transformers import BertTokenizer, BertModel
import deepspeed


class chatglm1_train(object):
    
    def __init__(self, data_args, model_args, prompt_column, 
                response_column, history_column, prefix, tokenizer, 
                task=False, department=False) -> None:
    
        self.data_args = data_args
        self.model_args = model_args
        self.prompt_column = prompt_column
        self.response_column = response_column
        self.history_column = history_column
        self.prefix = prefix
        self.tokenizer = tokenizer
        self.task = task
        self.sent = task
        self.department = department

        # 加载预训练的BERT模型和分词器
        self.bert_model_name = 'google-bert/bert-base-chinese'  #google-bert/bert-base-chinese from hugging face
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
        self.bert_model = BertModel.from_pretrained(self.bert_model_name).to("cuda")


    def __call__(self, examples):
        max_seq_length = self.data_args.max_source_length + self.data_args.max_target_length
        model_inputs = {
            "input_ids": [],
            "labels": [],
        }

        if self.task:
            model_inputs["task_id"] = []
            task_dict = json.load(open("datasets/task_dataset.json", "r"))
            task_dict = task_dict["str2id"]
        if self.sent:
            model_inputs["sent_rep"] = []


        for i in range(len(examples[self.prompt_column])):
            if examples[self.prompt_column][i] and examples[self.response_column][i]:
                query, answer = examples[self.prompt_column][i], examples[self.response_column][i]

                if self.history_column is None:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[self.history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

                prompt = self.prefix + prompt

                bert_prompt = prompt if len(prompt) < 510 else prompt[:510]
                bert_inputs = self.bert_tokenizer(bert_prompt, return_tensors='pt', truncation=True, padding=True)
                bert_inputs = {k: v.to("cuda") for k, v in bert_inputs.items()}     # 将输入张量移到 GPU
                with torch.no_grad():                    
                    bert_outputs = self.bert_model(**bert_inputs)
                bert_cls_embedding = bert_outputs.last_hidden_state[:, 0, :]
                bert_cls_embedding = bert_outputs.last_hidden_state[:, 0, :].cpu()  # 将 BERT 输出张量移回 CPU（如果后续不需要继续在 GPU 上操作）


                a_ids = self.tokenizer.encode(text=prompt, add_special_tokens=False)
                b_ids = self.tokenizer.encode(text=answer, add_special_tokens=False)

                if len(a_ids) > self.data_args.max_source_length - 1:
                    a_ids = a_ids[: self.data_args.max_source_length - 1]

                if len(b_ids) > self.data_args.max_target_length - 2:
                    b_ids = b_ids[: self.data_args.max_target_length - 2]

                input_ids = self.tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)
                input_ids.append(self.tokenizer.eos_token_id)   #lhy_add

                context_length = len(a_ids) #bloom

                mask_position = context_length - 1
                labels = [-100] * context_length + input_ids[mask_position+1:]
                
                pad_len = max_seq_length - len(input_ids)
                #input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
                #labels = labels + [self.tokenizer.pad_token_id] * pad_len

                if self.data_args.ignore_pad_token_for_loss:    #这里没有用处，之前没有做padding,等价于labels = labels
                    labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)

                if self.task:
                    task_id = task_dict[examples['task_dataset'][i]]
                    model_inputs["task_id"].append(task_id)
                    sent_rep = bert_cls_embedding[0].tolist()
                    model_inputs["sent_rep"].append(sent_rep)

        return model_inputs



class chatglm1_eval(object):
    
    def __init__(self, data_args, model_args, prompt_column, 
                response_column, history_column, prefix, tokenizer, 
                task=False, department=False) -> None:
        
        self.data_args = data_args
        self.model_args = model_args
        self.prompt_column = prompt_column
        self.response_column = response_column
        self.history_column = history_column
        self.prefix = prefix
        self.tokenizer = tokenizer
        self.task = task
        self.sent = task
        self.department = department

        # 加载预训练的BERT模型和分词器
        self.bert_model_name = '/root/lhy/model/bert-base-chinese'
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
        self.bert_model = BertModel.from_pretrained(self.bert_model_name).to("cuda")
        

    def __call__(self, examples):
    
        max_target_length = self.data_args.max_target_length
        inputs, targets = [], []

        if self.task:
            task_id = []
            task_dict = json.load(open("datasets/task_dataset.json", "r"))
            task_dict = task_dict["str2id"]
        if self.sent:
            sent_rep = []

        for i in range(len(examples[self.prompt_column])):
            if not examples[self.response_column][i]:
                targets.append("filled in !")
            else:
                targets.append(examples[self.response_column][i])

            if examples[self.prompt_column][i]:
                query = examples[self.prompt_column][i]
                if self.history_column is None or len(examples[self.history_column][i]) == 0:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[self.history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
                inputs.append(prompt)

            if self.task:
                bert_prompt = prompt if len(prompt) < 510 else prompt[:510]
                bert_inputs = self.bert_tokenizer(bert_prompt, return_tensors='pt', truncation=True, padding=True)
                bert_inputs = {k: v.to("cuda") for k, v in bert_inputs.items()}     # 将输入张量移到 GPU
                with torch.no_grad():
                    bert_outputs = self.bert_model(**bert_inputs)
                bert_cls_embedding = bert_outputs.last_hidden_state[:, 0, :]
                bert_cls_embedding = bert_outputs.last_hidden_state[:, 0, :].cpu()  # 将 BERT 输出张量移回 CPU（如果后续不需要继续在 GPU 上操作）


                task_id.append(task_dict[examples['task_dataset'][i]])
                sent_rep.append(bert_cls_embedding[0].tolist())

        inputs = [self.prefix + inp for inp in inputs]
        model_inputs = self.tokenizer(inputs,
                                    max_length=self.data_args.max_source_length,
                                    truncation=True,
                                    padding=True)
        labels = self.tokenizer(text_target=targets, max_length=max_target_length, truncation=True)

        if self.data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]

        if self.task:
            model_inputs["task_id"] = task_id
        if self.sent:
            model_inputs["sent_rep"] = sent_rep

        return model_inputs
        

