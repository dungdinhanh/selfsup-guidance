#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import hfai.client
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from ffrecord.torch import DataLoader
# from guided_diffusion.vision_images import *
from scripts_gdiff.selfsup.support.vision_images import *
from torch.utils.data.distributed import DistributedSampler
# from scripts_gdiff.selfsup.support import dist_util
# import hfai.nccl.distributed as dist

from guided_diffusion.script_util import create_gaussian_diffusion, diffusion_defaults
from guided_diffusion.resample import create_named_schedule_sampler, ScheduleSampler
from guided_diffusion.respace import SpacedDiffusion

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4096, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--image_size', default=64, type=int, help='Image size')

# additional configs:
parser.add_argument('--pretrained', default='', type=str,
                    help='path to simsiam pretrained checkpoint')
parser.add_argument('--lars', action='store_true',
                    help='Use LARS')
parser.add_argument("--save_folder", default="runs/linear_eval", type=str, help="Linear evaluation checkpoint")

best_acc1 = 0


def main():
    args = parser.parse_args()
    #
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
#
#
def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    save_folder = args.save_folder
    os.makedirs(save_folder, exist_ok=True)
    last_file = os.path.join(save_folder, "latest.pt")
    #
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args, flush=True):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        ip = os.environ['MASTER_IP']
        port = os.environ['MASTER_PORT']
        dist.init_process_group(backend=args.dist_backend, init_method=f'tcp://{ip}:{port}',
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
    # create model
    print("=> creating model '{}'".format(args.arch), flush=True)
    model = models.__dict__[args.arch]()
    # change first layer
    if args.image_size == 64 or args.image_size == 128:
        # Change first layer
        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        n = conv1.kernel_size[0] * conv1.kernel_size[1] * conv1.out_channels
        conv1.weight.data.normal_(0, math.sqrt(2. / n))
        model.conv1 = conv1

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        print(name)
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False


    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    # diffusion model
    args_dict = diffusion_defaults()
    args_input={
    'steps': args_dict['diffusion_steps'],
    'learn_sigma': args_dict['learn_sigma'],
    'sigma_small': False,
    'noise_schedule': "linear", #note
    'use_kl': args_dict['use_kl'],
    'predict_xstart': args_dict['predict_xstart'],
    'rescale_timesteps': args_dict['rescale_timesteps'],
    'rescale_learned_sigmas': args_dict['rescale_learned_sigmas'],
    'timestep_respacing': args_dict['timestep_respacing']}

    diffusion = create_gaussian_diffusion(**args_input)
    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)



    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained), flush=True )
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            # state_dict = checkpoint['state_dict']
            state_dict = checkpoint
            for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer
                if k.startswith('encoder.') and not k.startswith('encoder.fc'):
                    # remove prefix
                    state_dict[k[len("encoder."):]] = state_dict[k]
                if k.startswith('module.encoder.') and not k.startswith('module.encoder.fc'):
                    state_dict[k[len("module.encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            # print(state_dict)
            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            # print(msg.missing_keys)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(args.pretrained), flush=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained), flush=True)

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias

    optimizer = torch.optim.SGD(parameters, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.lars:
        print("=> use LARS optimizer.", flush=True)
        from apex.parallel.LARC import LARC
        optimizer = LARC(optimizer=optimizer, trust_coefficient=.001, clip=False)

    # optionally resume from a checkpoint
    # if args.resume:
    if os.path.isfile(last_file):
        print("=> loading checkpoint '{}'".format(last_file), flush=True)
        if args.gpu is None:
            checkpoint = torch.load(last_file)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(last_file, map_location=loc)
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        if args.gpu is not None:
            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(args.gpu)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        s_iter = checkpoint['s_iter']
        count= checkpoint['count']
        acc1 = checkpoint['acc1']
        acc5 = checkpoint['acc5']
        loss = checkpoint['loss']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(last_file, checkpoint['epoch']), flush=True)
    else:
        print("=> no checkpoint found at '{}'".format(last_file), flush=True)
        count = 0
        acc1 = 0.0
        acc5 = 0.0
        loss = 0.0
        s_iter = 0

    cudnn.benchmark = True

    # Data loading code
    # train_dataset/val_dataset needs to be changed later for augmentation
    train_dataset = ImageNetHFAug_la(args.image_size, random_crop=True, random_flip=False, split="train", classes=True, miniset=False)
    val_dataset = ImageNetHFAug_la(args.image_size, random_crop=False, random_flip=False, split="val", classes=True, miniset=False)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], # keep there for later use
                                     std=[0.229, 0.224, 0.225])
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None

    train_loader = train_dataset.loader(batch_size=args.batch_size, sampler=train_sampler, pin_memory=True, num_workers=8)
    val_loader = val_dataset.loader(batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=False)

    if args.evaluate:
        validate(val_loader, model, criterion, args, diffusion, schedule_sampler)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, diffusion, schedule_sampler, s_iter, count, acc1, acc5, loss,
              ngpus_per_node, last_file)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args, diffusion, schedule_sampler)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                's_iter': 0,
                'count': 0,
                'acc1': 0.0,
                'acc5': 0.0,
                'loss': 0.0,
            }, is_best, last_file)
            if epoch == args.start_epoch:
                sanity_check(model.state_dict(), args.pretrained)


def train(train_loader, model, criterion, optimizer, epoch, args, diffusion:SpacedDiffusion,
          schedule_sampler:ScheduleSampler, s_iter=0, count=0, acc1=0.0, acc5=0.0, loss=0.0,
          ngpus_per_node=8,last_file="checkpoint.pt"):
    assert diffusion is not None, "Diffusion is None"
    assert schedule_sampler is not None, "Schedule sampler is None"

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    if count != 0:
        losses.update(loss, count)
        top1.update(acc1, count)
        top5.update(acc5, count)

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        if i < s_iter:
            continue
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target['y'].cuda(args.gpu, non_blocking=True)

        t, _ = schedule_sampler.sample(images.shape[0], device=torch.device(f"cuda:{args.gpu}"))
        images = diffusion.q_sample(images, t)


        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        if (not args.multiprocessing_distributed and hfai.client.receive_suspend_command()) or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node and hfai.client.receive_suspend_command()):
            save_last_iter({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                's_iter': i+1,
                'count': top1.count,
                'acc1': top1.sum,
                'acc5': top5.sum,
                'loss': losses.sum,
            }, last_file)


def validate(val_loader, model, criterion, args, diffusion: SpacedDiffusion, schedule_sampler: ScheduleSampler):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target['y'].cuda(args.gpu, non_blocking=True)

            t, _ = schedule_sampler.sample(images.shape[0], device=torch.device(f"cuda:{args.gpu}"))
            images = diffusion.q_sample(images, t)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5), flush=True)

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    base_folder = os.path.dirname(filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(base_folder, 'model_best.pth.tar'))

def save_last_iter(state, filename):
    torch.save(state, filename)


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights), flush=True)
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'encoder.' + k[len('module.'):] \
            if k.startswith('module.') else 'encoder.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.", flush=True)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
