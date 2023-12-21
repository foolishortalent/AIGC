
from transformers import HfArgumentParser
from transformers import AutoTokenizer, AutoModel

import datasets

from peft import get_peft_model, LoraConfig, TaskType
from utils import get_data_collator
from training_arguments import MyTrainingArguments, MyTrainer


def build_model(training_args):
    print("#> Building model...")
    model = AutoModel.from_pretrained(
        "ZhipuAI/chatglm3-6b", load_in_8bit=True, trust_remote_code=True, device_map="auto"
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    # model.lm_head = CastOutputToFloat(model.lm_head)
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=training_args.lora_rank,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query_key_value"]
    )
    model = get_peft_model(model, "LuXun-lora-ch")

    print("#> Model built.")
    print("#> Total Trainable Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("#> Total Parameters:", sum(p.numel() for p in model.parameters()), "\n")

    return model


def main():
    # parse args
    training_args = HfArgumentParser(MyTrainingArguments).parse_args_into_dataclasses()[0]

    training_args.remove_unused_columns = False

    print("#> Loading dataset...")

    dataset = datasets.load_from_disk(training_args.dataset_path)
    dataset.set_format(
        type=dataset.format["type"],
        columns=list(dataset.features.keys()),
    )

    print("#> Dataset loaded.", "Total samples:", len(dataset), "\n")

    # build model

    model = build_model(training_args)
    tokenizer = AutoTokenizer.from_pretrained("ZhipuAI/chatglm3-6b", trust_remote_code=True)

    print("#> Start training...")
    # start train
    trainer = MyTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=get_data_collator(tokenizer),
    )
    trainer.train()
    model.save_pretrained(training_args.output_dir)

    print("#> Training finished.")
    print("#> Model saved to:", training_args.output_dir)


if __name__ == "__main__":
    main()