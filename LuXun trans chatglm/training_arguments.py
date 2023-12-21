# coding=utf-8
import sys

sys.path.append("./")
from dataclasses import dataclass, field
import os
from transformers import (

    TrainingArguments,
    Trainer,
)


@dataclass
class MyTrainingArguments(TrainingArguments):
    max_steps: int = field(default=5000)
    save_steps: int = field(default=1000)
    learning_rate: float = field(default=1e-4)
    fp16: bool = field(default=True)
    remove_unused_columns: bool = field(default=False)
    logging_steps: int = field(default=50)
    output_dir: str = field(default="luxun-lora")
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=2)
    dataset_path: str = field(default="lunxun-style-data/luxun")
    lora_rank: int = field(default=8)


import torch


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)

