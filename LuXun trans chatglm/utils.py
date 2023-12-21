from transformers import AutoTokenizer
import torch


# Completing tokens
def get_data_collator(tokenizer: AutoTokenizer):
    def data_collator(features: list) -> dict:
        len_ids = [len(feature["input_ids"]) for feature in features]
        longest = max(len_ids)
        input_ids = []
        labels_list = []
        for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
            ids = feature["input_ids"]
            seq_len = feature["seq_len"]
            labels = (
                [-100] * (longest-ids_l+seq_len) + ids[seq_len:]
            )
            ids = [tokenizer.pad_token_id] * (longest - ids_l) + ids
            _ids = torch.LongTensor(ids)
            labels_list.append(torch.LongTensor(labels))
            input_ids.append(_ids)
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels_list)
        return {
            "input_ids": input_ids,
            "labels": labels,
        }
    return data_collator


