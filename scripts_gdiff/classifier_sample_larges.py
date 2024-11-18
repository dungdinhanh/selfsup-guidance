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
from guided_diffusion.script_util import (
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
import hfai.client
from torchvision import utils
from scripts_gdiff.utils import *
import math

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
    base_folder = args.base_folder
    save_folder = os.path.join(
        base_folder,
        args.logdir,
        "logs",
    )
    rank = dist.get_rank()

    logger.configure(save_folder, rank=dist.get_rank())

    
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(os.path.join(base_folder, args.model_path), map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()


    # logger.log("loading classifier...")
    # classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    # classifier.load_state_dict(
    #     dist_util.load_state_dict(os.path.join(base_folder, args.classifier_path), map_location="cpu")
    # )
    # classifier.to(dist_util.dev())
    # if args.classifier_use_fp16:
    #     classifier.convert_to_fp16()
    # classifier.eval()

    # def cond_fn(x, t, y=None):
    #     assert y is not None
    #     with th.enable_grad():
    #         x_in = x.detach().requires_grad_(True)
    #         logits = classifier(x_in, t)
    #         log_probs = F.log_softmax(logits, dim=-1)
    #         selected = log_probs[range(len(logits)), y.view(-1)]
    #         return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    # logger.log("Looking for previous file")
    # checkpoint = os.path.join(output_images_folder, "samples_last.npz")
    # vis_images_folder = os.path.join(output_images_folder, "sample_images")
    # saved_images_folder = os.path.join(output_images_folder, "images")
    # os.makedirs(saved_images_folder, exist_ok=True)
    # os.makedirs(vis_images_folder, exist_ok=True)
    # final_file = os.path.join(output_images_folder,
    #                           f"samples_{args.num_samples}x{args.image_size}x{args.image_size}x3.npz")
    # if os.path.isfile(final_file):
    #     dist.barrier()
    #     logger.log("sampling complete")
    #     return
    # if os.path.isfile(checkpoint):
    #     npzfile = np.load(checkpoint)
    #     all_images = list(npzfile['arr_0'])
    #     all_labels = list(npzfile['arr_1'])
    # else:
    #     all_images = []
    #     all_labels = []
    # output_images_folder = os.path.join(base_folder, args.logdir, "reference")
    # output_raw_images = os.path.join(base_folder, args.logdir ,"images")
    # os.makedirs(output_images_folder, exist_ok=True)
    # os.makedirs(output_raw_images, exist_ok=True)

    output_folder_path = os.path.join(base_folder, args.logdir)
    sample_folder_dir = os.path.join(output_folder_path, f"images")
    reference_dir = os.path.join(output_folder_path, "reference")
    vis_images_folder = os.path.join(output_folder_path, "vis_images")
    os.makedirs(reference_dir, exist_ok=True)
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    list_png_files, max_index = get_png_files(sample_folder_dir)

    final_file = os.path.join(reference_dir,
                              f"samples_{args.num_samples}x{args.image_size}x{args.image_size}x3.npz")
    if os.path.isfile(final_file):
        dist.barrier()
        print("Sampling complete")
        dist.barrier()
        dist.destroy_process_group()
        return

    checkpoint = os.path.join(sample_folder_dir, "last_samples.npz")

    if os.path.isfile(checkpoint):
        ckpt = np.load(checkpoint)
        all_images = list(ckpt['arr_0'])
        all_labels = list(ckpt["arr_1"])
        if len(list_png_files) > 0:
            all_images, all_labels = compress_images_to_npz(sample_folder_dir, all_images, all_labels)
    else:
        all_images = []
        all_labels = []
        all_images, all_labels = compress_images_to_npz(sample_folder_dir, all_images, all_labels)

    no_png_files = len(all_images)
    if no_png_files >= args.num_samples:
        if rank == 0:
            print(f"Complete sampling {no_png_files} satisfying >= {args.num_samples}")
            # create_npz_from_sample_folder(os.path.join(base_folder, args.sample_dir), args.num_samples, args.image_size)
            arr = np.stack(all_images)
            arr = arr[: args.num_samples]
            shape_str = "x".join([str(x) for x in arr.shape])
            out_path = os.path.join(reference_dir, f"samples_{shape_str}.npz")
            # logger.log(f"saving to {out_path}")
            print(f"Saving to {out_path}")
            np.savez(out_path, arr)
            os.remove(checkpoint)
            print("Done.")
        dist.barrier()
        dist.destroy_process_group()
        return
    else:
        if rank == 0:
            # remove_prev_npz(args.sample_dir, args.num_samples, args.image_size)
            print("continue sampling")

    total_samples = int(math.ceil((args.num_samples - no_png_files) / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Already sampled {no_png_files}")
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = len(all_images)



    logger.log(f"Number of current images: {len(all_images)}")
    logger.log("sampling...")
    if args.image_size == 28:
        img_channels = 1
        num_class = 10
    else:
        img_channels = 3
        num_class = NUM_CLASSES
    current_samples = 0
    for _ in pbar:
        model_kwargs = {}
        classes = th.randint(
            low=0, high=num_class, size=(args.batch_size,), device=dist_util.dev()
        )
        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        samples = sample_fn(
            model_fn,
            (args.batch_size, img_channels, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=None,
            device=dist_util.dev(),
        )

        if args.save_imgs_for_visualization and dist.get_rank() == 0 and (
                len(all_images) // dist.get_world_size()) < 10:
            save_img_dir = vis_images_folder
            utils.save_image(
                samples.clamp(-1, 1),
                os.path.join(save_img_dir, "samples_{}.png".format(len(all_images))),
                nrow=4,
                normalize=True,
                range=(-1, 1),
            )

        samples = ((samples + 1) * 127.5).clamp(0, 255).to(th.uint8)
        samples = samples.permute(0, 2, 3, 1)
        samples = samples.contiguous().numpy()

        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}_c{classes[i]}.png")
        total += global_batch_size
        current_samples += global_batch_size
        dist.barrier()
        if current_samples >= 500 or total >= total_samples:
            if rank == 0:
                all_images, all_labels = compress_images_to_npz(sample_folder_dir, all_images, all_labels)
            current_samples = 0
            pass
        # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        # batch_images = [sample.cpu().numpy() for sample in gathered_samples]
        # all_images.extend(batch_images)
        # gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        # dist.all_gather(gathered_labels, classes)
        # batch_labels = [labels.cpu().numpy() for labels in gathered_labels]
        # all_labels.extend(batch_labels)
        # if dist.get_rank() == 0:
        #     if hfai.client.receive_suspend_command():
        #         print("Receive suspend - good luck next run ^^")
        #         hfai.client.go_suspend()
        #     logger.log(f"created {len(all_images) * args.batch_size} samples")
        #     np.savez(checkpoint, np.stack(all_images), np.stack(all_labels))

    # arr = np.concatenate(all_images, axis=0)
    # arr = arr[: args.num_samples]
    # label_arr = np.concatenate(all_labels, axis=0)
    # label_arr = label_arr[: args.num_samples]
    # if dist.get_rank() == 0:
    #     shape_str = "x".join([str(x) for x in arr.shape])
    #     out_path = os.path.join(sample_folder_dir, f"samples_{shape_str}.npz")
    #     logger.log(f"saving to {out_path}")
    #     np.savez(out_path, arr, label_arr)
    #     os.remove(checkpoint)

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

def compress_images_to_npz(sample_folder_dir, all_images=[], all_labels=[]):
    npz_file = os.path.join(sample_folder_dir, "sample_last.npz")
    list_png_files, _ = get_png_files(sample_folder_dir)
    no_png_files = len(list_png_files)
    if no_png_files <= 1:
        return all_images
    for i in range(no_png_files):
        image_png_path = os.path.join(sample_folder_dir, f"{list_png_files[i]}")
        try:
            # image_png_path = os.path.join(images_dir, f"{list_png_files[i]}")
            img = Image.open(image_png_path)
            img.verify()
        except(IOError, SyntaxError) as e:
            print(f'Bad file {image_png_path}')
            print(f'remove {image_png_path}')
            os.remove(image_png_path)
            continue
        file_name_woext = os.path.splitext(list_png_files[i])
        class_image = int(file_name_woext.split("_")[1][1:])
        sample_pil = Image.open(os.path.join(image_png_path))
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        all_images.append(sample_np)
        all_labels.append(class_image)
        os.remove(image_png_path)
    np_all_images = np.stack(all_images)
    np_all_labels = np.stack(all_labels)
    np.savez(npz_file, arr_0=np_all_images, arr_1=np_all_labels)
    return all_images, all_labels

if __name__ == "__main__":
    ngpus = th.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus, bind_numa=False)
