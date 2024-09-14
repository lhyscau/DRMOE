import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, default_data_collator
import argparse
import time
import os
os.environ['WANDB_DISABLED'] = 'true'

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()
        return item

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    args = parser.parse_args()

    #分布式 初始化进程组      
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)


    texts = []
    base_texts = [
        "这是第一个示例文本。",
        "这是第二个示例文本。",
        "这是第三个示例文本。",
        "这是第四个示例文本。",
    ]
    # 生成400个样本
    for i in range(20):
        for text in base_texts:
            texts.append(text)

    # 加载预训练模型和分词器
    model_name = "/root/lhy/model/bloomz-560m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # 创建数据集
    train_dset = TextDataset(texts, tokenizer)
    test_dset = TextDataset(texts, tokenizer)

    # 使用 DistributedDataParallel
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])    #分布式 将模型副本放到不同GPU中

    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss().to(device)

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir='./test_results',
        num_train_epochs=40,
        per_device_train_batch_size=8,  # 单GPU的批次大小
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        local_rank=args.local_rank,     #分布式 使用的GPU编号与进程一致
        remove_unused_columns=False
    )

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dset,
        eval_dataset=test_dset,
        data_collator=default_data_collator,
        optimizers=(optimizer, None),  # 指定优化器
    )
    start_time = time.time()
    # 开始训练
    trainer.train()
    end_time = time.time()
    # 打印训练花费的时间
    print(f"Training completed in {end_time - start_time} seconds")

if __name__ == "__main__":
    main()




# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 /root/lhy/MOELoRA-peft/test_trainer_version.py
