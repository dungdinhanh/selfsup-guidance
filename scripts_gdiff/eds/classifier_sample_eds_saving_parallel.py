"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

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
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
import datetime
from PIL import Image
from torchvision import utils
import hfai.client
import hfai.multiprocessing
import math
from scripts_gdiff.utils import *
import time


def main(local_rank):
    args = create_argparser().parse_args()

    dist_util.setup_dist(local_rank)

    if args.fix_seed:
        import random
        seed = 23333 + dist.get_rank()
        np.random.seed(seed)
        th.manual_seed(seed)  # CPU随机种子确定
        th.cuda.manual_seed(seed)  # GPU随机种子确定
        th.cuda.manual_seed_all(seed)  # 所有的GPU设置种子

        th.backends.cudnn.benchmark = False  # 模型卷积层预先优化关闭
        th.backends.cudnn.deterministic = True  # 确定为默认卷积算法

        random.seed(seed)
        np.random.seed(seed)

        os.environ['PYTHONHASHSEED'] = str(seed)

    save_folder = os.path.join(
        args.logdir,
        "logs",
    )

    logger.configure(save_folder, rank=dist.get_rank())

    output_images_folder = os.path.join(args.logdir, "reference")
    os.makedirs(output_images_folder, exist_ok=True)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()


    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            # return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale
            cond_grad = th.autograd.grad(selected.sum(), x_in)[0]

        guidance = {
            'gradient': cond_grad,
            'scale': args.classifier_scale
        }

        # a few lines of code to apply EDS
        if args.use_entropy_scale:
            with th.no_grad():
                probs = F.softmax(logits, dim=-1)  # (B, C)
                entropy = (-log_probs * probs).sum(dim=-1)  # (B,)
                entropy_scale = 1.0 / (entropy / np.log(NUM_CLASSES))  # (B,)
                entropy_scale = entropy_scale.reshape(-1, 1, 1, 1).repeat(1, *cond_grad[0].shape)
                guidance['scale'] = guidance['scale'] * entropy_scale

        return guidance


    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

        logger.log("Loading similarity matrix")
        sim_matrix = np.load(args.lsim_path)['arr_0']
        sim_matrix = torch.from_numpy(sim_matrix).to(dist_util.dev())

    logger.log("Looking for previous file")
    checkpoint = os.path.join(output_images_folder, "samples_last")
    os.makedirs(checkpoint, exist_ok=True)
    # checkpoint_temp = os.path.join(output_images_folder, "samples_temp.npz")

    final_file = os.path.join(output_images_folder,
                              f"samples_{args.num_samples}x{args.image_size}x{args.image_size}x3.npz")
    if os.path.isfile(final_file):
        dist.barrier()
        logger.log("sampling complete")
        return

    # Figure out how many samples we need to generate on each GPU and how many iterartions we need to run
    n = args.batch_size
    global_batch_size = n * dist.get_world_size()
    list_png_files, max_index = get_png_files(checkpoint)
    no_png_files = len(list_png_files)

    if no_png_files >= args.num_samples:
        if dist.get_rank() == 0:
            print(f"Complete sampling {no_png_files} satisfying >= {args.num_samples}")
            npz_file, state = create_npz_from_sample_folder(output_images_folder, args.num_samples, args.image_size)
            print("Done.")
            if not state:
                logger.log("The sampling has not been finished, some files are damaged. Please rerun")
        dist.barrier()
        dist.destroy_process_group()
        return
    total_samples = int(math.ceil((args.num_samples - no_png_files) / global_batch_size) * global_batch_size)

    logger.log(f"Number of current images: {no_png_files}")
    logger.log(f"Need sampling {total_samples}...")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    iterations = int(samples_needed_this_gpu // n)
    current_index = max_index

    current_num_images = no_png_files

    for i in range(iterations):
        model_kwargs = {}
        if args.specified_class is not None:
            classes = th.randint(
                low=int(args.specified_class), high=int(args.specified_class) + 1, size=(args.batch_size,),
                device=dist_util.dev()
            )
        else:
            classes = th.randint(
                low=0, high=args.num_classes, size=(args.batch_size,), device=dist_util.dev()
            )
        model_kwargs["y"] = classes

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        out = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )  # (B, 3, H, W)
        sample = out['sample']

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)  # (B, H, W, 3)
        sample = sample.contiguous()
        sample = sample.cpu().numpy()

        classes = classes.cpu().numpy()

        if dist.get_rank() == 0:
            if hfai.client.receive_suspend_command():
                if hfai.client.receive_suspend_command():
                    print("Receive suspend - good luck next run ^^")
                    hfai.client.go_suspend()
        # dist.barrier()

        for i, sample in enumerate(sample):
            index = i * dist.get_world_size() + dist.get_rank() + current_index
            Image.fromarray(sample).save(f"{checkpoint}/{index:06d}_{int(classes[i])}.png")
        current_index += global_batch_size
        current_num_images += global_batch_size
        logger.log(f"sampled {current_num_images}")

    if dist.get_rank() == 0:
        time.sleep(20)
        _, state = create_npz_from_sample_folder(output_images_folder, args.num_samples, args.image_size)
        if state:
            logger.log("sampling complete")
        else:
            logger.log("Sampling not complete, some files have been damaged. Please rerun sampling")
    dist.barrier()
    dist.destroy_process_group()
    return


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=50000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
        use_entropy_scale=True,
        num_classes=1000,
        save_imgs_for_visualization=True,
        fix_seed=False,
        specified_class=None,
        logdir=""
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser



if __name__ == "__main__":
    ngpus = th.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus, bind_numa=True)
