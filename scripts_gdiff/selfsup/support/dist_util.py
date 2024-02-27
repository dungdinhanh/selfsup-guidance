from guided_diffusion.dist_util import *


def load_simsiam(file_path="eval_models/simsiam_0099.pth.tar"):
    state_dict = load_state_dict(file_path, map_location="cpu")["state_dict"]
    for key in list(state_dict.keys()):
        first_string = str(key).split(".")[0]
        str_key = str(key)
        if first_string == "module":
            new_key = str_key.replace(f"{first_string}.", "")
        else:
            new_key = None

        if new_key is not None:
            state_dict[new_key] = state_dict.pop(str_key)
    return state_dict


def load_mocov2(file_path="eval_models/simsiam_0099.pth.tar"):
    state_dict = load_state_dict(file_path, map_location="cpu")["state_dict"]
    for key in list(state_dict.keys()):
        list_keys = str(key).split(".")
        first_string = list_keys[0]
        second_string = list_keys[1]
        str_key = str(key)
        if first_string == "module":
            new_key = str_key.replace(f"{first_string}.", "")
            if second_string == "encoder_q":
                new_key = new_key.replace(f"{second_string}.", "")
        else:
            new_key = None

        if new_key is not None:
            state_dict[new_key] = state_dict.pop(str_key)
    return state_dict

def load_mocov3(file_path="eval_models/simsiam_0099.pth.tar"):
    state_dict = load_state_dict(file_path, map_location="cpu")["state_dict"]
    new_state_dict = {}
    for key in list(state_dict.keys()):
        list_keys = str(key).split(".")
        first_string = list_keys[0]
        second_string = list_keys[1]
        str_key = str(key)
        if first_string == "module":
            new_key = str_key.replace(f"{first_string}.", "")
            if second_string == "base_encoder":
                new_key = new_key.replace(f"{second_string}.", "")
            else:
                new_key = None
        else:
            new_key = None

        if new_key is not None:
            new_state_dict[new_key] = state_dict.pop(str_key)
    return new_state_dict

def load_byol(file_path="eval_models/simsiam_0099.pth.tar"):
    state_dict = load_state_dict(file_path, map_location="cpu")["state_dict"]
    for key in list(state_dict.keys()):
        first_string = str(key).split(".")[0]
        str_key = str(key)
        if first_string == "backbone":
            new_key = str_key.replace(f"{first_string}.", "")
        else:
            new_key = None

        if new_key is not None:
            state_dict[new_key] = state_dict.pop(str_key)
    return state_dict
