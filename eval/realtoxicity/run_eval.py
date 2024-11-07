import argparse
import glob
import json
import os
import random
from collections import defaultdict
import ray
import torch
os.environ["VLLM_WORKER_MULTIPROC_METHOD"]="spawn"
# if "RAY_ADDRESS" in os.environ:
#     # 连接到现有集群
#     # ray.shutdown()
#     ray.init(address="auto")
# else:
#     # 如果没有现有集群，则启动新的集群
#     ray.shutdown()
#     ray.init(num_gpus=torch.cuda.device_count())
# import multiprocessing
# multiprocessing.set_start_method('spawn', force=True)

import vllm

from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from eval.utils import (
    generate_completions,
    load_hf_lm_and_tokenizer,
    query_openai_chat_model,
)
from eval.utils import dynamic_import_function 
import torch.nn.functional as F
import torch.quantization

import ray, torch

# ray.shutdown()

# ray.init(num_gpus=torch.cuda.device_count())


@torch.no_grad()
def score_generations(
    hate_classifier_tokenizer, hate_classifier_model, generations, batch_size=1
):
    classifications = []
    for i in trange(0, len(generations), batch_size):
        batch = generations[i : i + batch_size]
        inputs = hate_classifier_tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt"
        ).to(hate_classifier_model.device)
        outputs = hate_classifier_model(**inputs)
        classifications.extend(outputs.logits.argmax(dim=-1).tolist())
    return classifications


def main(args):
    torch.multiprocessing.set_start_method('spawn')
    random.seed(42)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load the testing data
    examples = []
    with open('challenge_prompts.jsonl', 'r') as f:
        for line in f.readlines():
            d = eval(line)
            examples.append(d['prompt'])

    # Generate the outputs
    if args.model_name_or_path:
        prompts = []
        prompts = examples

        if args.use_vllm:
            print("Loading vLLM model...")
            model = vllm.LLM(
                model=args.model_name_or_path,
                tokenizer=args.tokenizer_name_or_path if args.model_name_or_path else args.model_name_or_path,
                tokenizer_mode="slow" if args.use_slow_tokenizer else "auto",
            )
            print("out vllm")




            # model.get_tokenizer().add_tokens(["[s1]", "[s2]"])
            sampling_params = vllm.SamplingParams(
                temperature=0,  # greedy decoding
                max_tokens=20,  # maximum we can pass to roberta
            )
            print("out sample")
            outputs = model.generate(prompts, sampling_params)
            outputs = [it.outputs[0].text for it in outputs]
            print("not stuck because of vllm")
            print(outputs)



    with open(os.path.join(args.save_dir, "predictions.jsonl"), "w") as fout:
        for example in outputs:
            fout.write(json.dumps(example) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data/eval/toxigen"
    )
    parser.add_argument(
        "--toxic_path", 
        type=str,
        default="1"
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
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="If given, we will only use this many prompts per group. Default to 500 (half the available prompts).",
    )
    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (
        args.openai_engine is None
    ), "Either model_name_or_path or openai_engine should be specified."
    main(args)
