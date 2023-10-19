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
    create_classifier_and_diffusion_consistency,
)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict
import hfai.client
import numpy as np


def main(local_rank):
    args = create_argparser().parse_args()
    # save_model_folder = os.path.join(args.logdir, "models")
    os.makedirs(args.logdir, exist_ok=True)
    dist_util.setup_dist(local_rank)

    log_folder = os.path.join(
        args.logdir,
        "logs",
    )
    if dist.get_rank() == 0:
        logger.configure(log_folder, rank=dist.get_rank())
    else:
        logger.configure(rank=dist.get_rank())
    logger.log("creating model and diffusion...")

    model, diffusion = create_classifier_and_diffusion_consistency(
        **args_to_dict(args, classifier_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion
        )

    resume_step = 0

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
            logits, rep = model(sub_batch, timesteps=sub_t)
        return logits.detach(), rep.detach()


    data_iter = iter(data)
    list_reps = []
    list_logits = []
    count = 0
    while True:

        logits, rep = forward_backward_log(data_iter)
        count += rep.shape[0]
        list_reps.append(rep)
        list_logits.append(logits)
        print(count)
        if count >= args.num_samples:
            break
    reps = th.cat(list_reps).cpu().numpy()
    reps = reps[:args.num_samples]
    logits = th.cat(list_logits).cpu().numpy()
    logits = logits[:args.num_samples]
    np.savez(args.output_file, reps, logits)

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
        num_samples=50000
    )
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    ngpus = th.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus, bind_numa=False)
