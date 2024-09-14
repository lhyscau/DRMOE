# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset, DistributedSampler
# import torch.distributed as dist
# import argparse

# # 自定义随机数据集
# class RandomDataset(Dataset):
#     def __init__(self, num_samples, input_shape, num_classes):
#         self.num_samples = num_samples
#         self.input_shape = input_shape
#         self.num_classes = num_classes
#         self.data = torch.randn(num_samples, *input_shape)
#         self.targets = torch.randint(0, num_classes, (num_samples,))

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, idx):
#         return self.data[idx], self.targets[idx]

# # 简单的 MLP 模型
# class SimpleMLP(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(SimpleMLP, self).__init__()
#         self.fc1 = nn.Linear(input_size, 128)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
#     args = parser.parse_args()

#     # 初始化进程组
#     dist.init_process_group(backend='nccl')
#     torch.cuda.set_device(args.local_rank)

#     device = torch.device("cuda", args.local_rank)

#     # 创建随机数据集
#     input_shape = (100,)  # 输入形状，1D特征
#     num_classes = 10
#     num_samples = 60000  # 训练集样本数
#     num_test_samples = 10000  # 测试集样本数

#     train_dset = RandomDataset(num_samples, input_shape, num_classes)
#     test_dset = RandomDataset(num_test_samples, input_shape, num_classes)

#     train_sampler = DistributedSampler(train_dset)
#     train_loader = DataLoader(train_dset, batch_size=64, sampler=train_sampler)
#     test_loader = DataLoader(test_dset, batch_size=64, shuffle=False)

#     # 模型定义
#     model = SimpleMLP(input_shape[0], num_classes).to(device)
#     model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

#     optimizer = optim.AdamW(model.parameters(), lr=1e-3)
#     criterion = nn.CrossEntropyLoss().to(device)

#     # 训练循环
#     model.train()
#     for epoch in range(10):  # 训练10个epoch
#         train_sampler.set_epoch(epoch)
#         for batch_idx, (data, target) in enumerate(train_loader):
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             loss = criterion(output, target)
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#             if batch_idx % 100 == 0:
#                 print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

#     # 测试模型
#     model.eval()
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     print(f'Accuracy: {100. * correct / len(test_loader.dataset):.2f}%')

# if __name__ == "__main__":
#     main()

# #启动命令
# #CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 /root/lhy/MOELoRA-peft/test.py



import torch

def load_and_print_parameters(model_path):
    # 加载模型权重
    model_state_dict = torch.load(model_path, map_location='cpu')
    
    # 遍历所有参数
    for name, param in model_state_dict.items():
        print(f"Parameter name: {name}, Shape: {param.shape}")

# 示例权重文件路径
model_path = '/root/lhy/MOELoRA-peft/saved/moelora/checkpoint-2779/adapter_model.bin'  # 或者是 'path/to/your/weights.bin'

load_and_print_parameters(model_path)