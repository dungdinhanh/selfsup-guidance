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

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_data_imagenet_hfai
from guided_diffusion.resample import create_named_schedule_sampler
from scripts_gdiff.consistency.support.script_util_consistency import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict
import hfai.client
import numpy as np


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
    output_file = os.path.join(args.logdir, "reps.npz")
    if dist.get_rank() == 0:
        logger.configure(log_folder, rank=dist.get_rank())
    else:
        logger.configure(rank=dist.get_rank())
    logger.log("creating model and diffusion...")

    model, diffusion = create_classifier_and_diffusion(
        **args_to_dict(args, classifier_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion
        )

    resume_step = 0
    # 417('out', Sequential(
    #     (0): GroupNorm32(32, 512, eps=1e-05, affine=True)
    # (1): SiLU()
    # (2): AttentionPool2d(
    #     (qkv_proj): Conv1d(512, 1536, kernel_size=(1,), stride=(1,))
    # (c_proj): Conv1d(512, 1000, kernel_size=(1,), stride=(1,))
    # (attention): QKVAttention()
    # )
    # ))
    # 418('out.0', GroupNorm32(32, 512, eps=1e-05, affine=True))
    # 419('out.1', SiLU())
    # 420('out.2', AttentionPool2d(
    #     (qkv_proj): Conv1d(512, 1536, kernel_size=(1,), stride=(1,))
    # (c_proj): Conv1d(512, 1000, kernel_size=(1,), stride=(1,))
    # (attention): QKVAttention()
    # ))
    # 421('out.2.qkv_proj', Conv1d(512, 1536, kernel_size=(1,), stride=(1,)))
    # 422('out.2.c_proj', Conv1d(512, 1000, kernel_size=(1,), stride=(1,)))
    # 423('out.2.attention', QKVAttention())
    features = []
    for index, (module, name) in enumerate(list(zip(model.modules(), model.named_modules()))):

        if name[0] == "out.1":
            features.append(FeatureHook(module))
            print(name)
        elif name[0] == "out.2.attention":
            features.append(FeatureHook(module))
            print(name)



    model.load_state_dict(
        dist_util.load_state_dict("models/64x64_classifier.pt", map_location="cpu"))

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

    def forward_backward_log(data_loader, prefix="train"):
        model.eval()
        batch, extra = next(data_loader)
        labels = extra["y"].to(dist_util.dev())

        batch = batch.to(dist_util.dev())
        # Noisy images

        t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())

        for i, (sub_batch, sub_labels, sub_t) in enumerate(
            split_microbatches(args.microbatch, batch, labels, t)
        ):
            logits = model(sub_batch, timesteps=sub_t)
            rep = features[1].feature_list
        return logits.detach(), rep.detach()


    data_iter = iter(data)
    list_reps = []
    list_logits = []
    count = 0
    while True:

        logits, rep = forward_backward_log(data_iter)
        count += rep.shape[0]
        print(rep.shape)
        print(logits.shape)
        # list_reps.append(rep.cpu().numpy())
        list_logits.append(logits.cpu().numpy())
        print(count)
        if count >= args.num_samples:
            break
    # reps   = np.concatenate(list_reps, axis=0)
    # reps   = reps[:args.num_samples]
    logits = np.concatenate(list_logits, axis=0)
    logits = logits[:args.num_samples]
    np.savez(output_file, logits)

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
        logdir="eval_models/imn64",
        num_samples=500000
    )
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    ngpus = th.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus, bind_numa=False)
