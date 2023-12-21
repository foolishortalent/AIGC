import json
from tqdm import tqdm

import datasets
import transformers


def preprocess(tokenizer, config, example, max_seq_length):
    prompt = example["context"]
    target = example["target"]
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(target, max_length=max_seq_length, truncation=True, add_special_tokens=False)
    input_ids = prompt_ids + target_ids + [config.eos_token_id]
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}


def read_jsonl(path, max_seq_length, skip_overlength=False):
    model_name = "ZhipuAI/chatglm3-6b"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    config = transformers.AutoConfig.from_pretrained(
        model_name, trust_remote_code=True, device_map='auto')
    with open(path, "r") as f:
        for line in tqdm(f.readlines()):
            example = json.loads(line)
            feature = preprocess(tokenizer, config, example, max_seq_length)
            if skip_overlength and len(feature["input_ids"]) > max_seq_length:
                continue
            feature["input_ids"] = feature["input_ids"][:max_seq_length]
            yield feature


def main():
    jsonl_path = "lunxun-style-data/luxun_data.jsonl"
    save_path = "lunxun-style-data/luxun"
    max_seq_length = 500
    skip_overlength = False

    print("#> Tokenizing dataset...")
    print("#> Input path: {}".format(jsonl_path))
    print("#> Output path: {}".format(save_path))
    print("#> Max sequence length: {}".format(max_seq_length))
    print("#> Skip overlength: {}".format(skip_overlength))

    dataset = datasets.Dataset.from_generator(
        lambda: read_jsonl(jsonl_path, max_seq_length, skip_overlength)
    )
    dataset.save_to_disk(save_path)

    print("#> Tokenization finished!", "Total examples:", len(dataset))



if __name__ == "__main__":
    main()