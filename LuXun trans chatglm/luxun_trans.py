from transformers import AutoModel
import torch

from transformers import AutoTokenizer

from peft import PeftModel
import argparse


def generate(instruction, text):
    with torch.no_grad():
        input = f"指令：{instruction}\n语句：{text}\n答："
        ids = tokenizer.encode(input)
        input_ids = torch.LongTensor([ids]).cuda()
        output = peft_model.generate(
            input_ids=input_ids,
            max_length=500,
            do_sample=False,
            temperature=0.0,
            num_return_sequences=1
        )
        output = tokenizer.decode(output[0])
        answer = output.split("答：")[-1]
    return answer.strip()


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--base_model", type=str, default="ZhipuAI/chatglm3-6b")
    argparser.add_argument("--lora", type=str, default="luxun-lora")
    argparser.add_argument("--instruction", type=str, default="用鲁迅的积极风格改写，保持原来的意思")


    args = argparser.parse_args()

    model = AutoModel.from_pretrained(args.base_model, trust_remote_code=True, load_in_8bit=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    if args.lora == "":
        print("#> No lora model specified, using base model.")
        peft_model = model.eval()
    else:
        print("#> Using lora model:", args.lora)
        peft_model = PeftModel.from_pretrained(model, args.lora).eval()
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    
    
    # 一些例子
    texts = [
        "你好",
        "有多少人工，便有多少智能。",
        "落霞与孤鹜齐飞，秋水共长天一色。",
        "我去买几个橘子，你就站在这里，不要走动。",
        "学习计算机技术，是没有办法救中国的。",
        "我怎么样都起不了床，我觉得我可能是得了抑郁症吧。",
        "它是整个系统的支撑架构，连接处理器、内存、存储、显卡和外围端口等所有其他组件。",
        "古巴导弹危机涉及美国和苏联之间的僵局，因苏联在古巴设立核导弹基地而引发，而越南战争则是北方（由苏联支持）和南方（由美国支持）之间在印度支那持续的军事冲突。",
        "齿槽力矩是指旋转设备受到齿轮牙齿阻力时施加的扭矩。",
        "他的作品包括蒙娜丽莎和最后的晚餐，两者都被认为是杰作。",
        "滑铁卢战役发生在1815年6月18日，是拿破仑战争的最后一场重大战役。"
    ]

    for text in texts:
        print(text)
        print(generate(args.instruction, text), "\n")
    
