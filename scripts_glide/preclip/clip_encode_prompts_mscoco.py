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


    logdir = "runs/pretext2img/"

    logger.configure(logdir, rank=dist.get_rank())

    output_images_folder = os.path.join(logdir, "reference")
    os.makedirs(output_images_folder, exist_ok=True)




    # options_up = model_and_diffusion_defaults_upsampler()
    # options_up['timestep_respacing'] = 'fast27'
    # options_up['use_fp16'] = args.use_fp16
    # model_up, diffusion_up = create_model_and_diffusion(**options_up)
    # model_up.load_state_dict(load_checkpoint('upsample', th.device("cpu")))
    # model_up.to(dist_util.dev())
    # if args.use_fp16:
    #     model_up.convert_to_fp16()
    # model_up.eval()
    # print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))


    logger.log("loading clip...")
    clip_model = create_clip_model(device=dist_util.dev())
    clip_model.image_encoder.load_state_dict(load_checkpoint('clip/image-enc', th.device("cpu")))
    clip_model.text_encoder.load_state_dict(load_checkpoint('clip/text-enc', th.device("cpu")))
    clip_model.image_encoder.to(dist_util.dev())
    clip_model.text_encoder.to(dist_util.dev())
    # classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    # classifier.load_state_dict(
    #     dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    # )
    # classifier.to(dist_util.dev())
    # if args.classifier_use_fp16:
    #     classifier.convert_to_fp16()
    # classifier.eval()

    # cond_fn = clip_model.cond_fn
    # cond_fn = clip_model.cond_fn([prompt] * batch_size, guidance_scale)





    caption_loader = load_data_caption_hfai(split="train", batch_size=args.batch_size)

    caption_iter = iter(caption_loader)
    num_captions = 0
    list_z_t = []
    while num_captions < args.num_captions:
        prompts = next(caption_iter)
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


if __name__ == "__main__":
    ngpus = th.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus, bind_numa=False)