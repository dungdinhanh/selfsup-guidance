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
import torchvision.models as models
from glide_text2im.clip.model_creation import create_clip_model
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)

from off_guided_diffusion.script_util import (
    classifier_defaults,
    create_classifier,
    NUM_CLASSES
)

from scripts_gdiff.selfsup.support.script_util_ss import create_simsiam_selfsup, simsiam_defaults, create_mocov2_selfsup
from scripts_gdiff.selfsup.support.dist_util import load_simsiam, load_mocov2

import datetime
from PIL import Image
from torchvision import utils
import hfai.client
import hfai.multiprocessing

from datasets.coco_helper import load_data_caption, load_data_caption_hfai


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
    model, diffusion = create_model_and_diffusion(
        **options_model
    )
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
    assert args.classifier_type in ['finetune', 'resnet50', 'resnet101', 'simsiam', 'mocov2']
    logger.log("loading classifier...")
    if args.classifier_type == 'finetune':
        classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
        classifier.load_state_dict(
            dist_util.load_state_dict(args.classifier_path, map_location="cpu")
        )
        classifier.to(dist_util.dev())
        if args.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()
    elif args.classifier_type == 'simsiam':
        resnet_address = 'eval_models/simsiam_0099.pth.tar'
        resnet = create_simsiam_selfsup(**args_to_dict(args, simsiam_defaults().keys()))
        for param in resnet.parameters():
            param.required_grad = False
        resnet.load_state_dict(load_simsiam(resnet_address))
        resnet.to(dist_util.dev())
        resnet.eval()
        resnet.sampling = True
    elif args.classifier_type == 'mocov2':
        resnet_address = 'eval_models/moco_v2_800ep_pretrain.pth.tar'
        resnet = create_mocov2_selfsup(**args_to_dict(args, simsiam_defaults().keys()))
        for param in resnet.parameters():
            param.required_grad = False
        resnet.load_state_dict(load_mocov2(resnet_address))
        resnet.to(dist_util.dev())
        resnet.eval()
        resnet.sampling = True
    else:
        if args.classifier_type == 'resnet50':
            resnet_address = 'eval_models/resnet50-19c8e357.pth'
            resnet = models.resnet50()
        elif args.classifier_type == 'resnet101':
            resnet_address = 'eval_models/resnet101-5d3b4d8f.pth'
            resnet = models.resnet101()

        for param in resnet.parameters():
            param.required_grad = False
        resnet.load_state_dict(th.load(resnet_address))
        resnet.eval()
        resnet.cuda()

        # replace ReLU with Softplus activation function
        if (args.softplus_beta < np.inf):
            for name, module in resnet.named_children():
                if isinstance(module, th.nn.ReLU):
                    resnet._modules[name] = th.nn.Softplus(beta=args.softplus_beta)
                if name in ['layer1', 'layer2', 'layer3', 'layer4']:
                    for sub_name, sub_module in module.named_children():
                        if isinstance(sub_module, models.resnet.Bottleneck):
                            for subsub_name, subsub_module in sub_module.named_children():
                                if isinstance(subsub_module, th.nn.ReLU):
                                    resnet._modules[name]._modules[sub_name]._modules[subsub_name] = th.nn.Softplus(
                                        beta=args.softplus_beta)

    args.classifier_scale = float(args.classifier_scale)
    args.joint_temperature = float(args.joint_temperature)
    args.margin_temperature_discount = float(args.margin_temperature_discount)
    args.gamma_factor = float(args.gamma_factor)

    if dist.get_rank() == 0:
        print('args:', args)

    def model_fn(x, t, y=None, p_features=None, selected_indexes=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    # use off-the-shelf classifier gradient for guided sampling
    mean_imn = [0.485, 0.456, 0.406]
    std_imn = [0.229, 0.224, 0.225]

    # features loading
    features_folder = os.path.dirname(args.features)
    features_mean_sup_file = os.path.join(features_folder, f"reps3_mean_sup_closest{args.k_closest}_set.npz")
    if not os.path.isfile(features_mean_sup_file):
        features_file = np.load(args.features)
        # features_n = features_file['arr_0'].shape[0]
        features_p = features_file['arr_0']
        labels_associated = features_file['arr_1']
        features_p, labels_associated = get_mean_closest_sup(features_p, labels_associated)
        np.savez(features_mean_sup_file, features_p, labels_associated)
        logger.log(f"saving {features_mean_sup_file}")

    features_file = np.load(features_mean_sup_file)
    features_n = features_file['arr_0'].shape[0]
    features_p = features_file['arr_0']

    labels_associated = features_file['arr_1']

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

    caption_loader = load_data_caption_hfai(split="val", batch_size=args.batch_size)

    caption_iter = iter(caption_loader)
    while len(all_images) * args.batch_size < args.num_samples:
        prompts = next(caption_iter)
        while len(prompts) != args.batch_size:
            prompts = next(caption_iter)
        #

        # print(len(prompts))
        cond_fn = clip_model.cond_fn(prompts, guidance_scale)

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
        logdir=""
    )
    defaults.update(model_and_diffusion_defaults())
    # defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def similarity_match(pred, target, mean=False):
    pred_norm = th.nn.functional.normalize(pred, dim=1)
    target_norm = th.nn.functional.normalize(target, dim=1)
    # cosine_sim = (pred_norm * target_norm).sum(dim=1)
    cosine_sim = th.matmul(pred_norm, target_norm.T)
    loss = cosine_sim
    if mean:
        loss = loss.sum()
    return loss

def find_closest_set(mean_vector, vectors, k=5):
    n = vectors.shape[0]
    list_distances = []

    for i in range(n):
        # print(vectors[i])
        # print(mean_vector)
        distance = np.linalg.norm(vectors[i] - mean_vector)
        list_distances.append(distance)

    list_distances = np.asarray(list_distances)
    indexes_sorted = np.argsort(list_distances)
    return vectors[indexes_sorted[:k]], indexes_sorted[:k]

def get_mean_closest_sup(features_p, labels):
    n = features_p.shape[0]
    dict_p_classes = {}
    p_mean_vectors = []
    labels_vectors = []
    for i in range(n):
        class_label = int(labels[i])
        if class_label not in dict_p_classes:
            dict_p_classes[class_label] = []
        dict_p_classes[class_label].append(features_p[i])

    for key in dict_p_classes.keys():
        p_vectors = np.stack(dict_p_classes[key])
        p_mean = np.mean(p_vectors, axis=0)
        labels_vectors.append(np.asarray([key]))
        p_closest_vectors = p_mean
        p_mean_vectors.append(p_closest_vectors)

    return np.stack(p_mean_vectors), np.concatenate(labels_vectors)


if __name__ == "__main__":
    ngpus = th.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus, bind_numa=True)
