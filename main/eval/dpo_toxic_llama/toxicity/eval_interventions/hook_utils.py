"""
Utility functions for hooking.
"""
from functools import partial
import torch
import torch.nn.functional as F
import random


def get_intervene_vector(model, config):
    """
    Get vector according to specifications in :config:
    """

    def _get_mlp_w_out(_config):
        layer = _config["layer"]
        idx = _config["idx"]
        return model.transformer.h[layer].mlp.c_proj.weight[idx]

    def _get_mlp_w_in(_config):
        w_in_idx = _config["w_ins"][0]
        layer = w_in_idx[0]
        idx = w_in_idx[1]
        return model.transformer.h[layer].mlp.c_fc.weight[:, idx]

    def _get_toxic_probe(_config):
        return torch.load(_config["datapath"])['mlp.weight'][1]

    def _get_svd(_config):
        topk_sorted_score = _config["topk_sorted_score"]
        u_idx = _config["idx"]
        toxic_vector = torch.load(_config["datapath"])['mlp.weight'][1]
        return get_svd_u_vec(model, toxic_vector, topk_sorted_score, u_idx)

    def _get_random(_config):
        shape = model.transformer.h[0].mlp.c_proj.weight[0].shape
        device = model.device
        return torch.rand(shape).to(device)

    return {
        "mlp_w_out": _get_mlp_w_out,
        "mlp_w_in": _get_mlp_w_in,
        "toxic_probe": _get_toxic_probe,
        "svd": _get_svd,
        "random": _get_random,
    }[config["type"]](config)




def hook_subtract(model, config, toxic_probe=None):
    return model, []



def dont_hook(model, config, toxic_probe=None):
    return model, []
