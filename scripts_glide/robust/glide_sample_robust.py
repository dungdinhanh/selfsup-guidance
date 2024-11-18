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
    add_dict_to_argparser,
    args_to_dict,
)

from glide_text2im.clip.model_creation import create_clip_model_robust
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation_robust import (
    create_model_and_diffusion_prog,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)

import datetime
from PIL import Image
from torchvision import utils
import hfai.client
import hfai.multiprocessing

from datasets.coco_helper import load_data_caption, load_data_caption_hfai, load_data_caption_hfai_robust


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
    options_model = args_to_dict(args, model_and_diffusion_defaults().keys())
    options_model['use_fp16'] = args.use_fp16
    model, diffusion = create_model_and_diffusion_prog(
        **options_model
    )
    diffusion.osc = args.osc
    diffusion.eps = args.eps
    model.load_state_dict(
        load_checkpoint('base', th.device("cpu"))
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    logger.log('total base parameters', sum(x.numel() for x in model.parameters()))

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
    clip_model = create_clip_model_robust(device=dist_util.dev())
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


    logger.log("Looking for previous file")
    checkpoint = os.path.join(output_images_folder, "samples_last.npz")
    checkpoint_temp = os.path.join(output_images_folder, "samples_temp.npz")
    vis_images_folder = os.path.join(output_images_folder, "sample_images")
    os.makedirs(vis_images_folder, exist_ok=True)
    final_file = os.path.join(output_images_folder,
                              f"samples_{args.num_samples}x{args.image_size}x{args.image_size}x3.npz")
    if os.path.isfile(final_file):
        dist.barrier()
        logger.log("sampling complete")
        return
    if os.path.isfile(checkpoint):
        npzfile = np.load(checkpoint)
        all_images = list(npzfile['arr_0'])
    else:
        all_images = []
    logger.log(f"Number of current images: {len(all_images)}")
    logger.log("sampling...")

    guidance_scale = args.guidance_scale

    caption_loader = load_data_caption_hfai_robust(split="val", batch_size=args.batch_size)

    caption_iter = iter(caption_loader)
    while len(all_images) * args.batch_size < args.num_samples:
        prompts_set = next(caption_iter)
        while len(prompts_set[0]) != args.batch_size:
            prompts_set = next(caption_iter)
        #

        prompts, prompts_others = prompts_set[0], prompts_set[1]
        prompts_others = [list(a) for a in zip(*prompts_others)]
        cond_fn = clip_model.cond_fn(prompts, guidance_scale, prompts_others)

        tokens = model.tokenizer.encode_batch(prompts)
        tokens, mask = model.tokenizer.padded_tokens_and_mask_batch(
            tokens, options_model['text_ctx']
        )

        model_kwargs = dict(
            tokens=th.tensor(tokens, device=dist_util.dev()),
            mask=th.tensor(mask, dtype=th.bool, device=dist_util.dev()),
        )

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        model.del_cache()
        out = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )  # (B, 3, H, W)
        model.del_cache()
        sample = out

        if args.save_imgs_for_visualization and dist.get_rank() == 0 and (
                len(all_images) // dist.get_world_size()) < 10:
            save_img_dir = vis_images_folder
            utils.save_image(
                sample.clamp(-1, 1),
                os.path.join(save_img_dir, "samples_{}.png".format(len(all_images))),
                nrow=4,
                normalize=True,
                range=(-1, 1),
            )

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)  # (B, H, W, 3)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        if dist.get_rank() == 0:
            if hfai.client.receive_suspend_command():
                print("Receive suspend - good luck next run ^^")
                hfai.client.go_suspend()
            logger.log(f"created {len(all_images) * args.batch_size} samples")
            np.savez(checkpoint_temp, np.stack(all_images))
            if os.path.isfile(checkpoint):
                os.remove(checkpoint)
            os.rename(checkpoint_temp, checkpoint)

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]

    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(output_images_folder, f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)
        os.remove(checkpoint)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=50000,
        batch_size=16,
        use_ddim=False,
        guidance_scale=1.0,
        save_imgs_for_visualization=True,
        fix_seed=False,
        specified_class=None,
        logdir="",
        osc=0.2,
        eps=0.9,
    )
    defaults.update(model_and_diffusion_defaults())
    # defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser



if __name__ == "__main__":
    ngpus = th.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus, bind_numa=True)
