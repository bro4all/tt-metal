# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import re

from models.tt_transformers.tt.load_checkpoints import (
    convert_hf_qkv_to_meta_format,
    convert_meta_qkv_to_hf_format,
    map_hf_to_meta_keys,
    map_hf_to_meta_keys_vision_only,
    map_meta_to_hf_keys,
    split_hf_keys,
    standardize_hf_keys,
)


# NE MOZE DA SE MENJA
def standardize_hf_keys_multimodal(state_dict):
    all_keys = tuple(state_dict.keys())
    new_state_dict = {}
    for k in all_keys:
        if "model.visual." in k:
            new_state_dict[k.replace("model.visual.", "visual.")] = state_dict[k]
        elif "model.vision_tower.vision_model." in k:
            new_state_dict[k.replace("model.vision_tower.vision_model.", "visual.")] = state_dict[k]
        elif "model.language_model." in k:
            new_state_dict[k.replace("model.language_model.", "model.")] = state_dict[k]
        else:
            new_state_dict[k] = state_dict[k]

    # Standardize keys used in vision parts of Qwen2.5-VL
    state_dict = standardize_hf_keys(new_state_dict)
    replace_whole_name = lambda pattern, repl: lambda s: re.sub(rf"(^|\.)({pattern})($|\.)", rf"\1{repl}\3", s)
    output = {}
    for k, v in state_dict.items():
        k = replace_whole_name("qkv", "qkv_proj")(k)
        k = replace_whole_name("proj", "o_proj")(k)
        k = replace_whole_name("attn", "self_attn")(k)
        output[k] = v
    return output


# NE MOZE DA SE MENJA, jer koristi map_vision_hf_to_meta_keys koji je specifican za GEMMA3
def convert_vision_hf_to_meta(state_dict, head_dim):
    state_dict = split_hf_keys(state_dict)
    state_dict = map_vision_hf_to_meta_keys(state_dict, head_dim)

    return state_dict


# NE MOZE DA SE MENJA
def map_vision_hf_to_meta_keys_split_to_submodels(state_dict):
    vision_state_dict = dict()
    text_state_dict = dict()
    other_state_dict = dict()

    for k, v in state_dict.items():
        if k.startswith("model.vision_tower"):
            selected_dict = vision_state_dict
        elif k.startswith("model.language_model") or k.startswith("lm_head"):
            selected_dict = text_state_dict
        else:
            selected_dict = other_state_dict

        selected_dict[k] = v

    return vision_state_dict, text_state_dict, other_state_dict


# NE MOZE DA SE MENJA
def map_vision_hf_to_meta_keys(state_dict, head_dim):
    vision_state_dict, text_state_dict, other_state_dict = map_vision_hf_to_meta_keys_split_to_submodels(state_dict)

    text_state_dict = convert_hf_qkv_to_meta_format(text_state_dict, head_dim)
    text_state_dict = map_hf_to_meta_keys(text_state_dict)

    vision_state_dict = map_hf_to_meta_keys_vision_only(vision_state_dict)

    return {**vision_state_dict, **text_state_dict, **other_state_dict}


# VIDETI NA OSNOVU TESTOVA DA LI TREBA OBRISATI,
# AKO TESTOVI PROLAZE ONDA ZNACI DA TREBA OBRISATI, JER JE BESKORISNO SAMO SE IMPORTUJE OVO IZ TT_TRANSFORMERS
def convert_meta_to_hf(state_dict, head_dim):
    state_dict = convert_meta_qkv_to_hf_format(state_dict, head_dim)
    state_dict = map_meta_to_hf_keys(state_dict)
    return state_dict


# Funkcija map_vision_meta_to_hf_keys nigde nije deklarisana, tako da time
# convert_vision_meta_to_hf nece raditi, sto znaci da moze da se obrise.


def convert_vision_meta_to_hf(state_dict, head_dim):
    # state_dict = convert_meta_qkv_to_hf_format(state_dict, head_dim)
    state_dict = map_vision_meta_to_hf_keys(state_dict)
    # state_dict = None
    return state_dict
