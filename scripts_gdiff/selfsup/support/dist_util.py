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
