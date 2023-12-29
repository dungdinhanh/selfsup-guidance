import json
import numpy as np
import re


def read_json(file_name):
    with open(file_name) as f:
        data = json.load(f)
    return data

def convert_to_dict_label_fulltext(data: dict):
    new_data = []
    keys = data.keys()
    for key in keys:
        str_value = ' '.join(data[key])
        new_data.append(str_value)
    return new_data

import argparse
import os

import numpy as np
import torch as th
# import torch.distributed as dist
import hfai.nccl.distributed as dist
import torch.nn.functional as F
import hfai
from guided_diffusion import dist_util, logger
from eds_guided_diffusion.script_util import (
    NUM_CLASSES,
    add_dict_to_argparser,
    args_to_dict,
)

from glide_text2im.clip.model_creation import create_clip_model
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation_robust import (
    create_model_and_diffusion_infodeg,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)

import datetime
from PIL import Image
from torchvision import utils
import hfai.client
import hfai.multiprocessing

from datasets.coco_helper import load_data_caption, load_data_caption_hfai, load_data_caption_hfai_robust

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="runs/pretext2img")
parser.add_argument("--num_captions", type=int, default=10000 )
parser.add_argument("--batch_size", type=int, default=128)


def main(local_rank):
    args = parser.parse_args()

    dist_util.setup_dist(local_rank)


    logdir = "eval_models/pretext2img_idxtolabels/"

    logger.configure(logdir, rank=dist.get_rank())

    output_images_folder = os.path.join(logdir, "reference")
    os.makedirs(output_images_folder, exist_ok=True)



    logger.log("loading clip...")
    clip_model = create_clip_model(device=dist_util.dev())
    clip_model.image_encoder.load_state_dict(load_checkpoint('clip/image-enc', th.device("cpu")))
    clip_model.text_encoder.load_state_dict(load_checkpoint('clip/text-enc', th.device("cpu")))
    clip_model.image_encoder.to(dist_util.dev())
    clip_model.text_encoder.to(dist_util.dev())


    data = read_json('scripts_glide/admglide/imagenet1000_clsidx_to_labels.txt')
    caption_loader = convert_to_dict_label_fulltext(data)
    n = len(caption_loader)
    # caption_loader = load_data_caption_hfai(split="train", batch_size=args.batch_size)

    caption_iter = iter(caption_loader)
    num_captions = 0
    list_z_t = []
    while num_captions < n:
        prompts = [next(caption_iter)]
        z_t = clip_model.text_embeddings(prompts)
        z_t = z_t.detach().cpu().numpy()
        list_z_t.append(z_t)
        num_captions += len(prompts)
        print(f"Current number of captions: {num_captions}")

    list_z_t = np.concatenate(list_z_t)
    list_z_t = list_z_t[: args.num_captions]
    final_file = os.path.join(output_images_folder,
                              f"captions_{list_z_t.shape[0]}_{list_z_t.shape[1]}.npz")
    np.savez(final_file, list_z_t)
    dist.barrier()


if __name__ == '__main__':
    ngpus = th.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus, bind_numa=False)
