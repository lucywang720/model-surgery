import argparse
import glob
import json
import os
import random
from collections import defaultdict

import torch
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


def main(args):
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

        toxic_vector = torch.load(args.toxic_path)['mlp.weight'][1].cuda()
        v = torch.load(args.toxic_path)['mlp.weight'][1].cuda()


        print(toxic_vector.abs().mean())
        scores = []
        for layer in range(32):
            print(layer)
            mlp_outs = model.model.layers[layer].mlp.gate_proj.weight
            cos_sims = F.cosine_similarity(
                mlp_outs, v.unsqueeze(0), dim=1
            )
            cos_sims *= -1
            _topk = cos_sims.topk(k=11008) #100
            _values = [x.item() for x in _topk.values]
            _idxs = [x.item() for x in _topk.indices]
            topk = list(zip(_values, _idxs, [layer] * _topk.indices.shape[0]))
            scores.extend(topk)

        sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)

        with torch.no_grad():
            for x in sorted_scores[:4096*4]:
                model.model.layers[x[2]].mlp.gate_proj.weight[x[1]] += args.alpha * toxic_vector




        if args.save_model:
            model_to_save = model.module if hasattr(model, 'module') else model
            CONFIG_NAME = "config.json"
            WEIGHTS_NAME = "pytorch_model.bin"
            output_dir = args.save_dir
            os.makedirs(output_dir, exist_ok=True)
            output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
            output_config_file = os.path.join(output_dir, CONFIG_NAME)
            save_dict = model_to_save.state_dict()
            for key in list(save_dict.keys()):
                if "lora" in key:
                    del save_dict[key]
            torch.save(save_dict, output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_pretrained(output_dir)

    else:
        print("no model path input")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        "--alpha",
        type=float,
        default=1.0,
        help="If given, we will only use this many prompts per group. Default to 500 (half the available prompts).",
    )
    parser.add_argument(
        "--toxic_path", 
        type=str, 
        default="eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    parser.add_argument(
        "--save_model", 
        action="store_true", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    parser.add_argument(
        "--save_path", 
        type=str, 
        default="", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None), "Model_name_or_path should be specified."
    main(args)