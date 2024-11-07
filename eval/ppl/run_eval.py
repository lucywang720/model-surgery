import argparse
import glob
import json
import os
import random
from collections import defaultdict

import torch
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from eval.utils import (
    generate_completions,
    load_hf_lm_and_tokenizer,
    query_openai_chat_model,
)
from eval.utils import dynamic_import_function 
import pandas as pd
from datasets import Dataset
import pyarrow.parquet as pq
from torch.nn import CrossEntropyLoss
from datasets import load_dataset

def load_wikitext_data(file_path):
    """
    加载WikiText数据集的parquet文件
    
    Args:
        file_path: parquet文件的路径
    Returns:
        dataset: 加载后的数据集对象
    """
    # 方法1：使用pandas加载
    df = pd.read_parquet(file_path)
    dataset = Dataset.from_pandas(df)
    
    # 打印基本信息
    print(f"数据集大小: {len(dataset)} 条")
    print("\n数据集特征:")
    print(dataset.features)
    
    # 展示几个样本
    print("\n数据样例:")
    for i in range(min(3, len(dataset))):
        print(f"\n样本 {i+1}:")
        print(dataset[i])
    
    return dataset


def main(args):
    random.seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    datapath = 'dataset/wikitext'
    # print(datapath, data_config["dataname"], data_config["split"])
    # data = load_dataset(
    #     datapath, 'wikitext-2-raw-v1', split='test'
    # )
    data = load_dataset(
        "parquet",
        data_files="dataset/wikitext/wikitext-2-raw-v1/data/test-00000-of-00001-7231805191546d57.parquet",
        split="train"  # parquet文件默认会被当作训练集
    )
    data = [{"prompt": "\n\n".join(data["text"])}]

    prompts = [x["prompt"] for x in data]
    # print(prompts)
    
    # Generate the outputs
    if args.model_name_or_path:
        print("Loading model and tokenizer for generations...")
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.model_name_or_path,
            load_in_8bit=args.load_in_8bit,
            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
            gptq_model=args.gptq,
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )
        # new_line_token = tokenizer.encode("\n", add_special_tokens=False)[-1]
        # outputs = generate_completions(
        #     model=model,
        #     tokenizer=tokenizer,
        #     prompts=prompts,
        #     max_new_tokens=128,
        #     batch_size=args.eval_batch_size if args.eval_batch_size else 1,
        #     stop_id_sequences=[[new_line_token]],
        # )


    # encodings = data["prompt_input_ids"]
    tokenized = tokenizer(
        prompts,
        return_tensors="pt",
    )

    encodings = tokenized["input_ids"]
    # print(encodings.shape)

    max_length = 1024
    stride = 1024

    ending = encodings.shape[1]

    lls = []
    for i in tqdm(range(0, ending, stride)):
        # iterates through all 287644 tokens in wikitext test in windows of stride (usually max_length)
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.shape[1])

        trg_len = end_loc - i  # may be different from stride on last loop

        input_ids = encodings[:, begin_loc:end_loc].cuda()

        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()

            loss_fn = CrossEntropyLoss()
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
            )
            lls.append(loss * trg_len)
        ppl = torch.exp(torch.stack(lls).sum() / end_loc)



    


    performance = {'ppl': ppl.item()}
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        print(f"Average Toxicity: {ppl}")
        # for group in performance:
            # if group != "overall":
                # print(f"Average Toxicity for {group}: {performance[group]}")
        json.dump(performance, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data/eval/toxigen"
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="results/toxigen"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the tokenizer from here.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--openai_engine",
        type=str,
        default=None,
        help="if specified, we will use the OpenAI API to generate the predictions.",
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=1, help="batch size for evaluation."
    )
    parser.add_argument(
        "--classifier_batch_size",
        type=int,
        default=32,
        help="batch size to use for toxicity classifier.",
    )
    parser.add_argument(
        "--classifier_device",
        type=str,
        default="cuda",
        help="device to use for toxicity classifier.",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="If given, we're evaluating a 4-bit quantized GPTQ model.",
    )
    parser.add_argument(
        "--use_chat_format", 
        action="store_true", 
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function", 
        type=str, 
        default="eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="If given, we will use vLLM to generate the predictions - much faster.",
    )
    parser.add_argument(
        "--max_prompts_per_group",
        type=int,
        default=500,
        help="If given, we will only use this many prompts per group. Default to 500 (half the available prompts).",
    )
    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (
        args.openai_engine is None
    ), "Either model_name_or_path or openai_engine should be specified."
    main(args)
