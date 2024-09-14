# -*- encoding: utf-8 -*-
# here put the import lib
import os
import sys
import torch
import torch.distributed as dist

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PATH"] += ':/root/anaconda3/envs/moelora/bin/'  #解决ninja C++问题
from src.MLoRA.main import main
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from src.MLoRA.arguments import ModelArguments, DataTrainingArguments



if __name__ == "__main__":

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))    #原始保留

    # 初始化分布式训练环境
    # if 'LOCAL_RANK' in os.environ:
    #     local_rank = int(os.environ['LOCAL_RANK'])
    #     torch.cuda.set_device(local_rank)
    #     dist.init_process_group(backend='nccl')
    #     device = torch.device("cuda", local_rank)
    # else:
    #     local_rank = -1
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    # training_args.local_rank = local_rank

    # main(model_args, data_args, training_args)
    main(parser)  #原始保留

