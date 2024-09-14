# -*- encoding: utf-8 -*-
# here put the import lib
from dataclasses import dataclass
from typing import Dict, Sequence
import torch
import transformers

IGNORE_INDEX = -100


@dataclass
class LongestSequenceCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    task_flag: bool
    depart_flag: bool
    sent_flag: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        if self.task_flag:
            task_id = [instance["task_id"].tolist() for instance in instances]
            task_id = torch.LongTensor(task_id)
            # task_id = torch.tensor(task_id, dtype=torch.float32)    #bert的值很小，而不是原始论文的整形的task_id，因此这里需要使用float保持精度

            if self.depart_flag:    # if add the department and entity
                depart = [instance["depart"] for instance in instances]
                depart = torch.LongTensor(depart)

                entity = [instance["entity"] for instance in instances]
                entity = torch.stack(entity)
                entity = torch.LongTensor(entity)

                return dict(
                    input_ids=input_ids,
                    labels=labels,
                    task_id=task_id,
                    depart=depart,
                    entity=entity,
                )

            if self.sent_flag:
                sent_rep = [instance["sent_rep"].tolist() for instance in instances]
                sent_rep = torch.tensor(sent_rep, dtype=torch.float32)

                return dict(
                    input_ids=input_ids,
                    labels=labels,
                    task_id=task_id,
                    sent_rep=sent_rep
                )
            return dict(
                input_ids=input_ids,
                labels=labels,
                task_id=task_id,
            )

        return dict(
            input_ids=input_ids,
            labels=labels,
        )

