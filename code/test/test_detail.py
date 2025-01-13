import os
from tqdm import tqdm
import copy
import json
import fire
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoConfig, AddedToken
import torch
from loguru import logger
import copy
import sys


import argparse

from component.utils import ModelUtils
from component.template import template_dict
from utils_0 import read_task_in_jsonl, output_jsonl
def build_prompt(tokenizer, template, query, history, system=None):
    template_name = template.template_name
    system_format = template.system_format
    user_format = template.user_format
    assistant_format = template.assistant_format
    system = system if system is not None else template.system

    history.append({"role": 'user', 'message': query})
    input_ids = []

    # setting system information
    if system_format is not None:
        # system信息不为空
        if system is not None:
            system_text = system_format.format(content=system)
            input_ids = tokenizer.encode(system_text, add_special_tokens=False)
    # concat conversation
    for item in history:
        role, message = item['role'], item['message']
        if role == 'user':
            message = user_format.format(content=message, stop_token=tokenizer.eos_token)
        else:
            message = assistant_format.format(content=message, stop_token=tokenizer.eos_token)
        tokens = tokenizer.encode(message, add_special_tokens=False)
        input_ids += tokens
    input_ids = torch.tensor([input_ids], dtype=torch.long)

    return input_ids


def load_tokenizer(model_name_or_path):
    # config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    # 加载tokenzier
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False
        # llama不支持fast
        # use_fast=False if config.model_type == 'llama' else True
    )

    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    return tokenizer
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--start",type= int)
    parser.add_argument("--end",type= int)
    parser.add_argument("--i",type= int)
    parser.add_argument("--data_path",type= str)
    parser.add_argument("--model_path")
    args = parser.parse_args()
    template_name = 'llama2'

    data_path = args.data_path
    dataset = read_task_in_jsonl(data_path)[args.start:args.end]
    # model_name_or_path = os.path.join(args.models_folder, args.smell_type, "adapter_merged")
    # adapter_name_or_path = None
    
    # save disk space
    # model_name_or_path = "/data/zengyq/codellama/CodeQwen1.5-7B-Chat"
    model_name_or_path = args.model_path
    # adapter_name_or_path = os.path.join(args.models_folder, args.smell_type)
    adapter_name_or_path = None

    template = template_dict[template_name]
    # 是否使用4bit进行推理，能够节省很多显存，但效果可能会有一定的下降
    load_in_4bit = False
    # 生成超参配置
    max_new_tokens = 500
    top_p = 0.9
    temperature = 0.35
    repetition_penalty = 1.0

    # 加载模型
    logger.info(f'Loading model from: {model_name_or_path}')
    logger.info(f'adapter_name_or_path: {adapter_name_or_path}')
    model = ModelUtils.load_model(
        model_name_or_path,
        load_in_4bit=load_in_4bit,
        adapter_name_or_path=adapter_name_or_path,

    ).eval()
    tokenizer = load_tokenizer(model_name_or_path if adapter_name_or_path is None else adapter_name_or_path)

    if template.stop_word is None:
        template.stop_word = tokenizer.eos_token
    stop_token_id = tokenizer.convert_tokens_to_ids(template.stop_word)

    Output_json = []# [Conversation_ID Input Output Answer]
    #TODO:inference in BATCH
    BATCH_SIZE = 1
    model_name = args.model_path.split("/")[-1]
    output_folder = f"../../output/{model_name}"
    if os.path.exists(output_folder) == False:
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, f"{args.i}_inference.jsonl")
    for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
        # for a in dataset[i:i+1]:
        #     for key, value in a.items():
        #         print("键: {}，值: {}".format(key, value))
        
        input_prompt = [a["conversation"][0]["human"] for a in dataset[i:i+BATCH_SIZE]][0]
        
        try:
            input_ids = build_prompt(tokenizer, template, input_prompt, history = [], system=None).to(model.device)
            x = input_ids.size(1)
            input_ids = input_ids[:2000]
            outputs = model.generate(
                input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
                top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
                eos_token_id=stop_token_id
            )
            outputs = outputs.tolist()[0][len(input_ids[0]):]
            response = tokenizer.decode(outputs)    
            response = response.strip().replace(template.stop_word, "").strip()
        except KeyboardInterrupt:
            assert 1 == 2
        except Exception as e:
            print(e)
            x = -1
            response = "too long"
        
        # print("outputs:", outputs)
        
        Output_json.append({"label_output":dataset[i]["conversation"][0]["assistant"],"prediction":response, "input_ids_len":x})
    
        if i % 100 == 0 or i + 1 == len(dataset):
            output_jsonl(Output_json,output_path)

   
if __name__ == "__main__":
    main()