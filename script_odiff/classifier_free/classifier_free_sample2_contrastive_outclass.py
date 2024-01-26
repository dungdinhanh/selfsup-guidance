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
from guided_diffusion.script_util_classifier_free import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion_classifier_free2contrastive,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
    create_model,
    create_model_diffusion_unconditional
)
import torchvision.models as models
from scripts_gdiff.selfsup.support.script_util_ss import create_simsiam_selfsup, simsiam_defaults, create_mocov2_selfsup
from scripts_gdiff.selfsup.support.dist_util import load_simsiam, load_mocov2
import datetime
from PIL import Image
import hfai.client
from torchvision import utils

def center_crop_arr(images, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    y_size = images.shape[2]
    x_size = images.shape[3]
    crop_y = (y_size - image_size) // 2
    crop_x = (x_size - image_size) // 2
    return images[:, :, crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def custom_normalize(images, mean, std):
    # print(images.shape)
    # Check if the input tensor has the same number of channels as the mean and std
    if images.size(1) != len(mean) or images.size(1) != len(std):
        raise ValueError("The number of channels in the input tensor must match the length of mean and std.")
    images = images.to(th.float)
    # Normalize the tensor
    for c in range(images.size(1)):
        images[:, c, :, :] = (images[:, c, :, :] - mean[c]) / std[c]

    return images

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

    logger.log("creating unconditional model and diffusion...")

    model, diffusion = create_model_and_diffusion_classifier_free2contrastive(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()


    logger.log("loading conditional model...")
    uncond_model = create_model_diffusion_unconditional(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    uncond_model.load_state_dict(
        dist_util.load_state_dict(args.uncond_model_path, map_location="cpu")
    )
    uncond_model.to(dist_util.dev())
    if args.use_fp16:
        uncond_model.convert_to_fp16()
    uncond_model.eval()

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
        features_p, labels_associated = get_mean_closest_sup(features_p, labels_associated, k=args.k_closest)
        np.savez(features_mean_sup_file, features_p, labels_associated)
        logger.log(f"saving {features_mean_sup_file}")

    features_file = np.load(features_mean_sup_file)
    features_n = features_file['arr_0'].shape[0]
    features_p = features_file['arr_0']

    labels_associated = features_file['arr_1']

    def cond_contrastive_fn(inputs, t, y=None, p_features=None, selected_indexes=None, mask=None):
        assert y is not None
        with th.enable_grad():
            x = inputs[0]
            pred_xstart = inputs[1]
            # off-the-shelf ResNet guided
            pred_xstart = pred_xstart.detach().requires_grad_(True)

            pred_xstart_r = ((pred_xstart + 1) * 127.).clamp(0, 255)/255.0
            pred_xstart_r = center_crop_arr(pred_xstart_r, args.image_size)
            pred_xstart_r = custom_normalize(pred_xstart_r, mean_imn, std_imn)

            # p_features_selected = p_features[selected_indexes]
            # match2 = similarity_match(p_features_selected, p_features.detach())


            # resnet classifier
            p_x_0 = resnet(pred_xstart_r)
            match1 = similarity_match(p_x_0, p_features.detach())

            logits = match1
            # print(logits.shape)
            # exit(0)
            # logits_t = match2
            temperature1 = args.joint_temperature
            temperature2 = temperature1 * args.margin_temperature_discount
            numerator = th.exp(logits * temperature1)[range(len(logits)), selected_indexes.view(-1)].unsqueeze(1)
            denominator2 = th.exp(logits * temperature2).sum(1, keepdims=True)
            selected = th.log(numerator / denominator2)

            return  th.autograd.grad(selected.sum(), pred_xstart_r)[0] * args.classifier_scale
        # return uncond_model(x, t)
    diffusion.cond_contrastive_fn = cond_contrastive_fn
    diffusion.gamma_factor = args.gamma_factor


    def cond_fn(x, t, y=None, p_features=None, selected_indexes=None, mask=None):
        return uncond_model(x, t, None)

    def model_fn(x, t, y=None, p_features=None, selected_indexes=None, mask=None):
        assert y is not None
        return model(x, t,  y)

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
        all_labels = list(npzfile['arr_1'])
    else:
        all_images = []
        all_labels = []
    logger.log(f"Number of current images: {len(all_images)}")
    logger.log("sampling...")
    if args.image_size == 28:
        img_channels = 1
        num_class = 10
    else:
        img_channels = 3
        num_class = NUM_CLASSES
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        n = args.batch_size
        random_selected_indexes = np.random.randint(0, features_n, (n,), dtype=int)
        p_features = th.from_numpy(features_p).to(dist_util.dev())
        classes = th.from_numpy(labels_associated[random_selected_indexes]).to(dist_util.dev())

        mask = get_mask(labels_associated, random_selected_indexes)
        mask = th.from_numpy(mask).to(dist_util.dev())

        model_kwargs["y"] = classes
        model_kwargs["p_features"] = p_features
        model_kwargs["selected_indexes"] = th.from_numpy(random_selected_indexes).to(dist_util.dev())
        model_kwargs["mask"] = mask

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (args.batch_size, img_channels, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
            w_cond=args.cond_model_scale
        )


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
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        batch_images = [sample.cpu().numpy() for sample in gathered_samples]
        all_images.extend(batch_images)
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)
        batch_labels = [labels.cpu().numpy() for labels in gathered_labels]
        all_labels.extend(batch_labels)
        if dist.get_rank() == 0:
            if hfai.client.receive_suspend_command():
                print("Receive suspend - good luck next run ^^")
                hfai.client.go_suspend()
            logger.log(f"created {len(all_images) * args.batch_size} samples")
            np.savez(checkpoint_temp, np.stack(all_images), np.stack(all_labels))
            if os.path.isfile(checkpoint):
                os.remove(checkpoint)
            os.rename(checkpoint_temp, checkpoint)

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(output_images_folder, f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)
        os.remove(checkpoint)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=50000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        uncond_model_path="",
        cond_model_scale=1.0,
        save_imgs_for_visualization=True,
        fix_seed=False,
        specified_class=None,
        logdir="",
        num_classes=1000,
        features="eval_models/imn128_mocov2/reps3.npz",
        k_closest=5,
        classifier_path="",
        classifier_scale=1.0,
        classifier_type='resnet101',
        softplus_beta=np.inf,
        joint_temperature=1.0,
        margin_temperature_discount=1.0,
        gamma_factor=0.0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(simsiam_defaults())
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

def get_mean_closest_sup(features_p, labels, k=5):
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
        labels_vectors.append(np.asarray([key] * k))
        p_closest_vectors, indexes = find_closest_set(p_mean, p_vectors, k)
        p_mean_vectors.append(p_closest_vectors)

    return np.concatenate(p_mean_vectors), np.concatenate(labels_vectors)

def get_mask(labels_list, selected_indexes):
    mask = np.ones((selected_indexes.shape[0], labels_list.shape[0]))
    for i in range(selected_indexes.shape[0]):
        selected_index = selected_indexes[i]
        list_label_selected = np.where(labels_list == labels_list[selected_index])
        mask[i, list_label_selected] *= 0
        mask[i, selected_index] = 1
    return mask
    pass


if __name__ == "__main__":
    ngpus = th.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus, bind_numa=False)
