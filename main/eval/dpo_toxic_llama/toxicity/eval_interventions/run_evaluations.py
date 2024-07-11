"""
Evaluation Module for interventions
"""

from typing import Dict

import os
import copy
import torch
import sys
sys.path.append("/home/jxshi/dpo_toxic_llama")
from toxicity.eval_interventions.eval_utils import (
    load_model,
    load_data,
    tokenize,
    get_intervene_name,
    pretty_print_results,
)
from toxicity.eval_interventions.generate_funcs import (
    generate_default,
    get_prompts,
    get_gold,
)
from toxicity.eval_interventions.metric_funcs import (
    run_f1,
    run_perplexity,
    run_perspective_api,
    run_dummy,
)
from toxicity.eval_interventions.hook_utils import (
    dont_hook,
    hook_subtract,
)
from constants import (
    ROOT_DIR,
    PROFANITY,
    SEXUALLY_EXPLICIT,
    IDENTITY_ATTACK,
    THREAT,
    INSULT,
    SEVERE_TOXICITY,
    TOXICITY,
    PERSPECTIVE_API_ATTRIBUTES as ATTRIBUTES,
)
from utils import verbose_print, VERBOSE

DATA_DIR = os.path.join(ROOT_DIR, "data/intervene_data")
CKPT_DIR = os.path.join(ROOT_DIR, "checkpoints")


GENERATE_FUNCS = {
    "get_prompts": get_prompts,
    "get_gold": get_gold,
}
METRIC_FUNCS = {
    "f1": run_f1,
    "perplexity": run_perplexity,
    "dummy": run_dummy,
    "perspective_api": run_perspective_api,
}
HOOK_FUNCS = {
    "subtraction": hook_subtract,
}
UNHOOK_FUNCS = {}


def generate(model, data, intervene_config):
    """
    Test intervention on a specific metric.
    """
    return GENERATE_FUNCS.get(intervene_config["method"], generate_default)(
        model, data, intervene_config["params"]
    )


def run_metric(
    metric_type,
    model,
    data_obj,
    intervene_results: Dict[str, torch.LongTensor],
    config,
    intervene_config
):
    """
    Calculate specific metric.

    :intervene_results: Mapping from intervention specification to a tensor
        of shape [data_size, max_prompt_len + max_new_tokens]
    """
    return METRIC_FUNCS[metric_type](
        model,
        data_obj,
        intervene_results,
        config,
        intervene_config
    )


def hook_model(model, config, toxic_probe=None):
    """
    Hook model.
    """
    return HOOK_FUNCS.get(config["method"], dont_hook)(model, config["params"], toxic_probe=None)


def unhook_model(model, hooks):
    """
    Remove hooks in the model.
    """
    for hook in hooks:
        hook.remove()


def _eval_intervene(
    model, tokenizer, model_config, intervene_config, metric_configs
):
    """
    Evaluation intervention on set of metrics.
    """
    assert "method" in intervene_config
    intervene_config["params"]["device"] = model_config["device"]

    results = {}
    # toxic_probe = torch.load('/cluster/home2/whq/PPLM_llama/32_new_data_toxic_classifier_head_epoch_8.pt')['mlp.weight'][1]
    toxic_probe = None
    for _metric_conf in metric_configs:
        metric_type = _metric_conf["metric"]
        intervene_config["params"]["max_new_tokens"] = None

        verbose_print(f"Evaluating {metric_type}")
        data = _metric_conf["tokenized"]

        intervene_config["params"]["hook_timesteps"] = -1
        if metric_type == "perplexity":
            intervene_config["params"]["hook_timesteps"] = 0

        _, hooks = hook_model(model, intervene_config, toxic_probe)

        generations = {}
        do_generate = _metric_conf["generate"]
        if do_generate:

            intervene_config["params"]["max_new_tokens"] = _metric_conf[
                "max_new_tokens"
            ]
            intervene_config["params"]["batch_size"] = model_config[
                "batch_size"
            ]
            generations = generate(model, data, intervene_config)
            for gen in generations["pred_text"][:30]:
                verbose_print(gen)
            # print(generations["pred_text"])
            # exit(0)

        # print("metric_type", metric_type)
        results[metric_type] = run_metric(
            metric_type,
            model,
            data,
            generations,
            _metric_conf.get("params"),
            intervene_config
        )
        print("metric_type", metric_type, results[metric_type])
        unhook_model(model, hooks)
    return results


def unroll_intervene(configs):
    """
    Unroll any nested configurations.
    """
    # print(configs)
    # exit(0)
    configs = [configs[-1]]
    unrolled = []
    for _config in configs:
        method = _config["method"]
        if method != "subtraction":
            unrolled.append(_config)
            continue

        params = _config["params"]
        scales = params.pop("scales", [])
        if len(scales) < 1:
            raise RuntimeError("Missing scale value?")

        subtract_sets = params.pop("subtract_from", [])
        if len(subtract_sets) < 1:
            raise RuntimeError("Missing subtract_from value?")

        for scale in scales:
            for subtract_set in subtract_sets:
                config_copy = copy.deepcopy(_config)
                config_copy["params"]["scale"] = scale
                config_copy["params"]["subtract_from"] = subtract_set
                unrolled.append(config_copy)

    return unrolled


def tokenize_data(tokenizer, config):
    """
    Tokenize all data beforehand.
    """
    metric_configs = config["metrics"]

    tokenized_data = {}
    for _metric_conf in metric_configs:
        datapath = _metric_conf["datapath"]
        if datapath in tokenized_data:
            _metric_conf["tokenized"] = tokenized_data[datapath]
            continue

        data = load_data(_metric_conf)
        tokenized_data[datapath] = tokenize(tokenizer, data, _metric_conf)
        _metric_conf["tokenized"] = tokenized_data[datapath]


def run_eval(config):
    """
    Run eval!
    """
    model_config = config["model"]
    metric_configs = config["metrics"]
    interventions = config["interventions"]

    assert len(metric_configs) == len(
        list(set([x["metric"] for x in metric_configs]))
    ), "Mismatch -- you likely specified the same metric twice!"

    model, tokenizer = load_model(model_config)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # tokenizer.pad_token_id = 50256
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.tokenizer = tokenizer

    # Tokenize all data beforehand.
    for _metric_conf in metric_configs:
        if "params" not in _metric_conf:
            _metric_conf["params"] = {}
        _metric_conf["params"]["pad_token_id"] = tokenizer.pad_token_id
        _metric_conf["params"]["batch_size"] = model_config["batch_size"]
        _metric_conf["params"]["device"] = model_config["device"]

    tokenize_data(tokenizer, config)

    interventions = unroll_intervene(interventions)
    print("?", interventions)
    results = {}
    for intervene_config in interventions:

        intervene_name = get_intervene_name(intervene_config)
        verbose_print(f"  Evaluating intervention {intervene_name}")
        results[intervene_name] = _eval_intervene(
            model, tokenizer, model_config, intervene_config, metric_configs
        )
        pretty_print_results(results)
    return results

def main(args):
    """ Driver """
    config = {
        "model": {
            "model_or_path": args.model_path,
            "tokenizer": args.model_path,
            "batch_size": args.batch_size,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "metrics": [
            {
                "datapath": args.data_dir,
                "metric": "perspective_api",
                "max_prompt_size": 512,
                "max_new_tokens": args.max_token,
                "generate": True,
                "params": {"attributes": ATTRIBUTES, "include_prompt": False},
            },
            {
                "datapath": "data/wikitext",
                "dataname": "wikitext-2-raw-v1",
                "split": "test",
                "metric": "perplexity",
                "generate": False,
            },
        ],
        "interventions": [
            {"method": "noop", "params": {}},
        ],
    }
    results = run_eval(config)
    print("Final Results:")
    pretty_print_results(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="/cluster/data0/whq/data/eval/gsm/"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=""
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16, 
    )
    parser.add_argument(
        "--max_token", 
        type=int, 
        default=20, 
    )
    args = parser.parse_args()  
    main(args)
