#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
import pickle
import json
import logging
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append("..")
sys.path.append("../..")
sys.path.append("./")
os.environ['WANDB_DISABLED'] = 'true'
# from resources.Qwen.modeling_qwen2 import Qwen2ForCausalLM# from resources.gpt2.modeling_gpt2 import GPT2LMHeadModel# import resources.modeling_chatglm# from modeling_modeling_qwen2 import Qwen2ForCausalLM# from ...resources import modeling_chatglm, modeling_qwen2

from resources.modeling_gpt2 import GPT2LMHeadModel
from resources.modeling_bloom import BloomForCausalLM
import torch
from torch.utils.data import DataLoader, DistributedSampler
import time


import jieba
import numpy as np
import transformers
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_chinese import Rouge
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    set_seed,
)

from results.evaluate import func
from src.MLoRA.trainer_seq2seq import Seq2SeqTrainer
from src.MLoRA.peft import PeftModel, TaskType, get_peft_model
from src.MLoRA.peft import LoraConfig, AdaLoraConfig
from src.MLoRA.peft import MMOELoraConfigS
from src.data_processor.chatglm import chatglm1_train, chatglm1_eval
# from src.data_processor.chatglm2 import chatglm2_train, chatglm2_eval# from transformers import set_seed
from src.data_processor.collator import LongestSequenceCollator
import random

logger = logging.getLogger(__name__)
from accelerate import Accelerator

def seed_torch(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    

def main(parser):
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):  #原始
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 16 and sys.argv[15].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[15]))
    elif len(sys.argv) == 3 and sys.argv[2].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[2]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    accelerator = Accelerator()
    training_args.batched_training = data_args.batched_training # for batched training

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    # set_seed(training_args.seed)
    seed_torch()

    # Load dataset
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file      #datasets/train.json
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file    #datasets/test.json
        extension = data_args.test_file.split(".")[-1]

    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    print("raw_datasets: ", raw_datasets)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )
    config.pre_seq_len = model_args.pre_seq_len
    config.prefix_projection = model_args.prefix_projection

    print(model_args.pre_seq_len)
    print(model_args.prefix_projection)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    tokenizer.cls_token = tokenizer.eos_token
    tokenizer.sep_token = tokenizer.eos_token

    model = BloomForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    ).half().cuda() 

    if training_args.do_train:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    if model_args.peft_path is not None:
        logger.info("Peft from pre-trained model")
        # Resume_training
        if training_args.resume_from_checkpoint is not None:
            model = PeftModel.from_pretrained(model, model_args.peft_path, is_trainable=True)
        else:
            model = PeftModel.from_pretrained(model, model_args.peft_path, is_trainable=False)
    else:
        logger.info("Init new peft model")
        target_modules = model_args.trainable.split(',')
        modules_to_save = model_args.modules_to_save.split(',') if model_args.modules_to_save!="null" else None
        lora_rank = model_args.lora_rank
        lora_dropout = model_args.lora_dropout
        lora_alpha = model_args.lora_alpha
        print(target_modules)

        kwargs = {}
        if model_args.lora_name == "adalora":
            TargetLoraConfig = AdaLoraConfig
            task_type = TaskType.CAUSAL_LM
        elif model_args.lora_name == "moelora":
            TargetLoraConfig = MMOELoraConfigS
            kwargs = {
                  "task_num": model_args.task_num,
                  "task_embedding_dim": model_args.task_embedding_dim,
                  "expert_num": model_args.expert_num,
                  "sent_linear_dim": model_args.sent_linear_dim
                  }
            task_type = TaskType.CAUSAL_LMS
        else:
            TargetLoraConfig = LoraConfig
            task_type = TaskType.CAUSAL_LM
        
        peft_config = TargetLoraConfig(
            task_type=task_type,
            target_modules=target_modules,
            inference_mode=False,
            r=lora_rank, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            modules_to_save=modules_to_save,
            **kwargs
        )
        model = get_peft_model(model, peft_config)
    
    model.to(accelerator.device)
    training_args.dataloader_num_workers = accelerator.num_processes
    model.print_trainable_parameters()
    training_args.ddp_find_unused_parameters = False

    task_flag = False   # flag whether generate task_id from dataset
    depart_flag = False  # flag whether use the department and entity
    if (model_args.lora_name == "moelora"):
        task_flag = True


    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names

    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names 
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    prompt_column = data_args.prompt_column
    response_column = data_args.response_column
    history_column = data_args.history_column
    
    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length

    def print_dataset_example(example):
        print("input_ids: ",example["input_ids"])
        print("inputs: ", tokenizer.decode(example["input_ids"]))
        print("label_ids: ", example["labels"])

    preprocess_function_train = chatglm1_train(data_args, model_args, prompt_column,
                                            response_column, history_column, prefix,
                                            tokenizer, task_flag, depart_flag)
    preprocess_function_eval = chatglm1_eval(data_args, model_args, prompt_column,
                                                response_column, history_column, prefix,
                                                tokenizer, task_flag, depart_flag)


    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function_train,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on train dataset",
            )
            print_dataset_example(train_dataset[0])
            print_dataset_example(train_dataset[1])
        train_dataset.set_format("torch")

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on validation dataset",
            )
            print_dataset_example(eval_dataset[0])
            print_dataset_example(eval_dataset[1])
        eval_dataset.set_format("torch")

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on prediction dataset",
            )
            print_dataset_example(predict_dataset[0])
            print_dataset_example(predict_dataset[1])
        predict_dataset.set_format("torch")

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    
    if training_args.do_train:
        data_collator = LongestSequenceCollator(tokenizer, task_flag, depart_flag, True)
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=None,
            padding=False
        )

    def compute_metrics(eval_preds):
        # if accelerator.is_local_main_process:
        if accelerator.is_main_process:
            preds, labels = eval_preds
            # preds = np.argmax(preds, axis=2)
            preds = torch.from_numpy(preds) #from np to tensor
            labels = torch.from_numpy(labels)
            predictions = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            predictions = [pred.strip() for pred in predictions]

            # 将-100替换为pad_token_id
            labels[labels == -100] = tokenizer.pad_token_id
            labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            labels = [label.strip() for label in labels]

            list_test_samples = []
            tp = "datasets/test.json"
            with open(tp, "r", encoding="utf-8") as f:
                for line in f:
                    line = json.loads(line)
                    list_test_samples.append(line)

            assert len(labels) == len(list_test_samples)
            pp = os.path.join("results/pred", f"test_predictions.json")
            
            with open(pp, "w", encoding="utf-8") as writer:
                for idx, (p, l) in enumerate(zip(predictions, labels)):
                    samp = list_test_samples[idx]  # 获取测试样本
                    samp["target"] = p  # 将目标值替换为预测值
                    res = json.dumps(samp, ensure_ascii=False)
                    writer.write(f"{res}\n")
            
            return func(pp = pp, tp = tp)
        else:
            return {}


    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )
    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        save_prefixencoder=model_args.pre_seq_len is not None
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_predict:
        logger.info("*** Predict ***")

        list_test_samples = []
        with open(data_args.test_file, "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                list_test_samples.append(line)

        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_new_tokens=data_args.max_target_length,
            do_sample=True,
            top_p=0.7,
            temperature=0.95,
        )
        metrics = predict_results.metrics
        print(metrics)
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                predict_results.label_ids[predict_results.label_ids == -100] = tokenizer.pad_token_id
                labels = tokenizer.batch_decode(
                    predict_results.label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                labels = [label.strip() for label in labels]
                assert len(labels) == len(list_test_samples)

                output_prediction_file = os.path.join(training_args.output_dir, "test_predictions.json")

                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    for idx, (p, l) in enumerate(zip(predictions, labels)):
                        samp = list_test_samples[idx]       #list_test_samples是test.json中所有样本的一个列表
                        samp["target"] = p      #直接将答案替换为预测值
                        res = json.dumps(samp, ensure_ascii=False)
                        writer.write(f"{res}\n")

    return results



if __name__ == "__main__":
    main()
