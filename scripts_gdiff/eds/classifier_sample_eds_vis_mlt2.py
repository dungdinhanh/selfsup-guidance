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
from eds_guided_diffusion.script_util_mlt import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion_cdiv,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
import datetime
from PIL import Image
from torchvision import utils
import hfai.client
import hfai.multiprocessing


def main(local_rank):
    args = create_argparser().parse_args()

    dist_util.setup_dist(local_rank)

    if args.fix_seed:
        import random
        seed = 7200 + dist.get_rank()
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
    model, diffusion = create_model_and_diffusion_cdiv(
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


    def model_fn(x, t, y=None, cls=None):
        assert y is not None
        save_img_dir = vis_images_folder
        if True:
            np_y = y.detach().cpu().numpy()
            print(np_y)
        utils.save_image(
            x.clamp(-1, 1),
            os.path.join(save_img_dir, "samples_{}.png".format(t[0]+1)),
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )
        return model(x, t, cls if args.class_cond else None)

    logger.log("Looking for previous file")

    vis_images_folder = os.path.join(output_images_folder, "sample_images")
    os.makedirs(vis_images_folder, exist_ok=True)


    logger.log(f"Number of current images: 0")
    logger.log("sampling...")

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

    save_img_dir = vis_images_folder
    utils.save_image(
        sample.clamp(-1, 1),
        os.path.join(save_img_dir, "samples_{}.png".format(0)),
        nrow=4,
        normalize=True,
        range=(-1, 1),
    )
    dist.barrier()
    logger.log("sampling complete")


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
    hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus, bind_numa=False)
