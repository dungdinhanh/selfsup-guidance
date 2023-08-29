"""
Train a noised image self-supervised classifier on ImageNet.
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
from scripts_gdiff.selfsup.support.image_datasets import load_data_imagenet_hfai_aug
from scripts_gdiff.selfsup.support.resample_ss import create_named_schedule_sampler_ext
from scripts_gdiff.selfsup.support.script_util_ss import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    simsiam_and_diffusion_defaults,
    create_simsiam_and_diffusion,
)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict
import hfai.client


def main(local_rank):
    args = create_argparser().parse_args()
    save_model_folder = os.path.join(args.logdir, "models")
    os.makedirs(save_model_folder, exist_ok=True)
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

    model, diffusion = create_simsiam_and_diffusion(
        **args_to_dict(args, simsiam_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    if args.noised:
        schedule_sampler = create_named_schedule_sampler_ext(
            args.schedule_sampler, diffusion
        )
        # schedule_sampler.range = [args.min, args.max]

    resume_step = 0

    if args.resume_checkpoint:
        if args.resume_checkpoint == "simsiam":
            logger.log(
                f"loading model from simsiam: eval_models/simsiam_0099.pth.tar"
            )

            model.load_state_dict(
                dist_util.load_simsiam(
                    "eval_models/simsiam_0099.pth.tar"
                )
            )
        else:
            resume_step = parse_resume_step_from_filename(args.resume_checkpoint)

            logger.log(
                f"loading model from checkpoint: {args.resume_checkpoint}... at {resume_step} step"
            )
            model.load_state_dict(
                dist_util.load_state_dict(
                    args.resume_checkpoint
                )
            )



    model.eval()
    # Needed for creating correct EMAs and fp16 parameters.
    dist_util.sync_params(model.parameters())

    mp_trainer = MixedPrecisionTrainer(
        model=model, use_fp16=False, initial_lg_loss_scale=16.0
    )

    model = DDP(
        model,
        device_ids=[dist_util.dev()],
        # output_device=dist_util.dev(),
        broadcast_buffers=False,
        # bucket_cap_mb=128,
        find_unused_parameters=False,
    )

    logger.log("creating data loader...")
    # data = load_data(
    #     data_dir=args.data_dir,
    #     batch_size=args.batch_size,
    #     image_size=args.image_size,
    #     class_cond=True,
    #     random_crop=True,
    # )
    data = load_data_imagenet_hfai_aug(train=True, image_size=args.image_size,
                                   batch_size=args.batch_size, random_crop=True)


    logger.log("training classifier model...")

    def forward_backward_log(data_loader, prefix="train"):
        batch1, _ = next(data_loader)
        batch2, _ = next(data_loader)
        # labels = extra["y"].to(dist_util.dev())

        batch1 = batch1.to(dist_util.dev())
        batch2 = batch2.to(dist_util.dev())
        # Noisy images

        t1 = th.zeros((batch1.shape[0],), device=dist_util.dev(), dtype=th.long)
        t1 = t1 + 200
        t2 = t1 + 4
        batch1 = diffusion.q_sample(batch1, t1)
        batch2 = diffusion.q_sample(batch2, t2)


        for i, (sub_batch1, sub_batch2, sub_t1, sub_t2) in enumerate(
            split_microbatches(args.microbatch, batch1, batch2, t1, t2)
        ):
            # p1, z1 = model(sub_batch1, timesteps=sub_t1)
            # # loss = F.cross_entropy(logits, sub_labels, reduction="none")
            # p2, z2 = model(sub_batch2, timesteps=sub_t2)
            p1, p2, z1, z2 = model(sub_batch1, sub_batch2)
            loss1 = similarity_loss(p1, z2, False)
            loss2 = similarity_loss(p2, z1, False)

            loss = (loss1 + loss2) * 1/2

            losses = {}
            losses[f"{prefix}_loss"] = loss.detach()
            losses[f"{prefix}_loss1"] = loss1.detach()
            losses[f"{prefix}_loss2"] = loss2.detach()

            log_loss_dict(diffusion, sub_t1, losses)
            del losses
            loss = 0.5 * (loss1.mean() + loss2.mean())
            print(f"{sub_t1[0]} || {sub_t2[0]} || {loss}")
    data_iter = iter(data)
    for step in range(args.iterations - 0):
        logger.logkv("step", step + 0)
        logger.logkv(
            "samples",
            (step + 0 + 1) * args.batch_size * dist.get_world_size(),
        )
        forward_backward_log(data_iter)

    dist.barrier()


def similarity_loss(pred, target, mean=True):
    pred_norm = th.nn.functional.normalize(pred, dim=1)
    target_norm = th.nn.functional.normalize(target, dim=1)
    cosine_sim = -(pred_norm * target_norm).sum(dim=1)
    loss = cosine_sim
    if mean:
        loss = loss.mean()
    return loss


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step, model_folder="runs", latest=False):
    if dist.get_rank() == 0:
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(model_folder, f"model{step:06d}.pt"),
        )
        th.save(opt.state_dict(), os.path.join(model_folder, f"opt{step:06d}.pt"))

def save_model_latest(mp_trainer, opt, step, model_folder="runs"):
    if dist.get_rank() == 0:
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(model_folder, "latest.pt"),
        )
        th.save({'opt': opt.state_dict(),
                 'step': step}, os.path.join(model_folder, "optlatest.pt"))


def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


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
        schedule_sampler="uniform-2-steps",
        resume_checkpoint="",
        pretrained_cls="",
        log_interval=500,
        eval_interval=5,
        save_interval=25000,
        logdir="runs",
        min=0,
        max=100
    )
    defaults.update(simsiam_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    ngpus = th.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus, bind_numa=False)
