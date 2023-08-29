from guided_diffusion.image_datasets import *
from scripts_gdiff.selfsup.support.vision_images import *


def load_data_imagenet_hfai(
    *,
    train=True,
    batch_size,
    image_size,
    random_crop=False,
    random_flip=True,
    class_cond=True,
        miniset=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    # if not data_dir:
    #     raise ValueError("unspecified data directory")
    # all_files = _list_image_files_recursively(data_dir)
    # classes = None
    # if class_cond:
    #     # Assume classes are the first part of the filename,
    #     # before an underscore.
    #     class_names = [bf.basename(path).split("_")[0] for path in all_files]
    #     sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
    #     classes = [sorted_classes[x] for x in class_names]
    # dataset = ImageDataset(
    #     image_size,
    #     all_files,
    #     classes=classes,
    #     shard=MPI.COMM_WORLD.Get_rank(),
    #     num_shards=MPI.COMM_WORLD.Get_size(),
    #     random_crop=random_crop,
    #     random_flip=random_flip,
    # )
    if train:
        dataset = ImageNetHF2Views(image_size, random_crop=random_crop, random_flip=random_flip, split='train', classes=class_cond,
                             miniset=miniset)
    else:
        dataset = ImageNetHF2Views(image_size, random_crop=random_crop, random_flip=random_flip, split='val', classes=class_cond,
                             miniset=miniset)
    # if deterministic:
    #     loader = DataLoader(
    #         dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
    #     )
    # else:
    #     loader = DataLoader(
    #         dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
    #     )
    data_sampler = DistributedSampler(dataset, shuffle=True)
    loader = dataset.loader(batch_size, num_workers=8, sampler=data_sampler, pin_memory=True)
    while True:
        yield from loader # put all items of loader into list and concat all list infinitely

def load_data_imagenet_hfai_3views2imgs(
    *,
    train=True,
    batch_size,
    image_size,
    random_crop=False,
    random_flip=True,
    class_cond=True,
        miniset=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    # if not data_dir:
    #     raise ValueError("unspecified data directory")
    # all_files = _list_image_files_recursively(data_dir)
    # classes = None
    # if class_cond:
    #     # Assume classes are the first part of the filename,
    #     # before an underscore.
    #     class_names = [bf.basename(path).split("_")[0] for path in all_files]
    #     sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
    #     classes = [sorted_classes[x] for x in class_names]
    # dataset = ImageDataset(
    #     image_size,
    #     all_files,
    #     classes=classes,
    #     shard=MPI.COMM_WORLD.Get_rank(),
    #     num_shards=MPI.COMM_WORLD.Get_size(),
    #     random_crop=random_crop,
    #     random_flip=random_flip,
    # )
    if train:
        dataset = ImageNetHF3Views2Imgs(image_size, random_crop=random_crop, random_flip=random_flip, split='train', classes=class_cond,
                             miniset=miniset)
    else:
        dataset = ImageNetHF3Views2Imgs(image_size, random_crop=random_crop, random_flip=random_flip, split='val', classes=class_cond,
                             miniset=miniset)
    # if deterministic:
    #     loader = DataLoader(
    #         dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
    #     )
    # else:
    #     loader = DataLoader(
    #         dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
    #     )
    data_sampler = DistributedSampler(dataset, shuffle=True)
    loader = dataset.loader(batch_size, num_workers=8, sampler=data_sampler, pin_memory=True)
    while True:
        yield from loader # put all items of loader into list and concat all list infinitely

def load_data_imagenet_hfai_aug(
    *,
    train=True,
    batch_size,
    image_size,
    random_crop=False,
    random_flip=True,
    class_cond=True,
        miniset=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    # if not data_dir:
    #     raise ValueError("unspecified data directory")
    # all_files = _list_image_files_recursively(data_dir)
    # classes = None
    # if class_cond:
    #     # Assume classes are the first part of the filename,
    #     # before an underscore.
    #     class_names = [bf.basename(path).split("_")[0] for path in all_files]
    #     sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
    #     classes = [sorted_classes[x] for x in class_names]
    # dataset = ImageDataset(
    #     image_size,
    #     all_files,
    #     classes=classes,
    #     shard=MPI.COMM_WORLD.Get_rank(),
    #     num_shards=MPI.COMM_WORLD.Get_size(),
    #     random_crop=random_crop,
    #     random_flip=random_flip,
    # )
    if train:
        dataset = ImageNetHFAug(image_size, random_crop=random_crop, random_flip=random_flip, split='train', classes=class_cond,
                             miniset=miniset)
    else:
        dataset = ImageNetHFAug(image_size, random_crop=random_crop, random_flip=random_flip, split='val', classes=class_cond,
                             miniset=miniset)
    # if deterministic:
    #     loader = DataLoader(
    #         dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
    #     )
    # else:
    #     loader = DataLoader(
    #         dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
    #     )
    data_sampler = DistributedSampler(dataset, shuffle=True)
    loader = dataset.loader(batch_size, num_workers=8, sampler=data_sampler, pin_memory=True)
    while True:
        yield from loader # put all items of loader into list and concat all list infinitely