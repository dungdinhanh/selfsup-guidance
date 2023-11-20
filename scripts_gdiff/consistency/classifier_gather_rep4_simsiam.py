"""
Train a noised image classifier on ImageNet.
"""
# import hfai_env
# hfai_env.set_env('dbg')
import argparse
import os
import datetime

import blobfile as bf
import torch as th
# import torch.distributed as dist
import hfai.nccl.distributed as dist
import torch.nn.functional as F
# from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from hfai.nn.parallel import DistributedDataParallel as DDP
import hfai
from torch.optim import AdamW

from guided_diffusion import logger
from scripts_gdiff.selfsup.support import dist_util
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_data_imagenet_hfai
from guided_diffusion.resample import create_named_schedule_sampler
from scripts_gdiff.selfsup.support.script_util_ss import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_simsiam_and_diffusion,
    simsiam_and_diffusion_defaults
)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict
import hfai.client
import numpy as np
import torchvision.transforms as transforms

def center_crop_arr(images, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    y_size = images.shape[2]
    x_size = images.shape[3]
    crop_y = (y_size - image_size) // 2
    crop_x = (x_size - image_size) // 2
    return images[:, crop_y : crop_y + image_size, crop_x : crop_x + image_size]

def custom_normalize(images, mean, std):
    # Check if the input tensor has the same number of channels as the mean and std
    if images.size(1) != len(mean) or images.size(1) != len(std):
        raise ValueError("The number of channels in the input tensor must match the length of mean and std.")
    images = images.to(th.float)
    # Normalize the tensor
    for c in range(images.size(1)):
        images[:, c, :, :] = (images[:, c, :, :] - mean[c]) / std[c]

    return images

class FeatureHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.feature_list = []
        self.device_list = []

    def hook_fn(self, module, input, output):
        if th.is_tensor(input):
            device = input.get_device()
        elif isinstance(input, tuple):
            device = input[0].get_device()
        elif isinstance(input, list):
            print(input)
            exit(0)
        self.device_list = device
        self.feature_list = output

    def close(self):
        self.hook.remove()

def main(local_rank):
    args = create_argparser().parse_args()
    # save_model_folder = os.path.join(args.logdir, "models")
    os.makedirs(args.logdir, exist_ok=True)
    dist_util.setup_dist(local_rank)

    log_folder = os.path.join(
        args.logdir,
        "logs",
    )
    output_file = os.path.join(args.logdir, "reps3.npz")
    if dist.get_rank() == 0:
        logger.configure(log_folder, rank=dist.get_rank())
    else:
        logger.configure(rank=dist.get_rank())
    logger.log("creating model and diffusion...")

    model, diffusion = create_simsiam_and_diffusion(
        **args_to_dict(args, simsiam_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion
        )

    resume_step = 0


    model.load_state_dict(
        dist_util.load_simsiam(args.p_classifier))

    # Needed for creating correct EMAs and fp16 parameters.
    dist_util.sync_params(model.parameters())


    model = DDP(
        model,
        device_ids=[dist_util.dev()],
        # output_device=dist_util.dev(),
        broadcast_buffers=False,
        # bucket_cap_mb=128,
        find_unused_parameters=False,
    )

    logger.log("creating data loader...")

    data = load_data_imagenet_hfai(train=True, image_size=args.image_size,
                                   batch_size=args.batch_size, random_crop=True)
    if args.val_data_dir:
        val_data = load_data_imagenet_hfai(train=False, image_size=args.image_size,
                                           batch_size=args.batch_size, random_crop=True)
    else:
        val_data = None

    logger.log("training classifier model...")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    augmentation = [
        normalize]
    transform_act = transforms.Compose(augmentation)
    mean_imn = [0.485, 0.456, 0.406]
    std_imn = [0.229, 0.224, 0.225]
    def forward_backward_log(data_loader, prefix="train"):
        model.eval()
        batch, extra = next(data_loader)
        labels = extra["y"].to(dist_util.dev())

        # batch = batch.to(dist_util.dev())
        # Noisy images

        t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())

        for i, (sub_batch, sub_labels, sub_t) in enumerate(
            split_microbatches(args.microbatch, batch, labels, t)
        ):
            sub_batch = ((sub_batch + 1) * 127.5).clamp(0, 255)/255.0
            # sub_batch = sub_batch.permute(0, 2, 3, 1)
            # sub_batch = sub_batch.contiguous().cpu().numpy()
            sub_batch = center_crop_arr(sub_batch, args.image_size)
            sub_batch = custom_normalize(sub_batch, mean_imn, std_imn)
            sub_batch = sub_batch.to(dist_util.dev())
            p_logits, z_logits = model.module.forward_1view(sub_batch.detach())
        return p_logits.detach(), z_logits.detach(), labels.detach()


    data_iter = iter(data)
    list_reps = []
    list_p_logits = []
    list_z_logits = []
    list_labels = []
    count = 0
    while True:

        p_logits, z_logits, labels = forward_backward_log(data_iter)
        count += p_logits.shape[0]

        # list_reps.append(rep.cpu().numpy())
        list_p_logits.append(p_logits.cpu().numpy())
        list_z_logits.append(z_logits.cpu().numpy())
        list_labels.append(labels.cpu().numpy())
        print(count, flush=True)
        if count >= args.num_samples:
            break
    # reps   = np.concatenate(list_reps, axis=0)
    # reps   = reps[:args.num_samples]
    p_logits = np.concatenate(list_p_logits, axis=0)
    p_logits = p_logits[:args.num_samples]

    z_logits = np.concatenate(list_z_logits, axis=0)
    z_logits = z_logits[:args.num_samples]

    labels = np.concatenate(list_labels, axis=0)
    labels = labels[:args.num_samples]

    np.savez(output_file, p_logits, z_logits, labels)

    dist.barrier()


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def create_argparser():
    defaults = dict(
        data_dir="",
        val_data_dir="",
        noised=True,
        iterations=150000,
        lr=3e-4,
        weight_decay=0.0,
        anneal_lr=False,
        batch_size=4,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=500,
        eval_interval=5,
        save_interval=25000,
        logdir="eval_models/imn256_ss/",
        num_samples=500000,
        p_classifier="models/64x64_classifier.pt",
        image_size=128
    )
    defaults.update(simsiam_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    ngpus = th.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus, bind_numa=True)
