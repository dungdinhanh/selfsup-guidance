#!/bin/bash

export NCCL_P2P_DISABLE=1

TRAIN_FLAGS="--iterations 300000 --anneal_lr True --batch_size 60 --lr 6e-4 --save_interval 10000 --weight_decay 0.2"
CLASSIFIER_FLAGS="--image_size 64 --classifier_attention_resolutions 32,16,8 --classifier_depth 4 \
--classifier_width 128 --classifier_pool attention --classifier_resblock_updown True \
 --classifier_use_scale_shift_norm True"

cmd="cd .."
#echo ${cmd}
#eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}


cmd="python scripts_gdiff/selfsup/classifier_train_selfsup.py --data_dir path/to/imagenet --logdir runs/selfsup_training/scratch $TRAIN_FLAGS $CLASSIFIER_FLAGS"
echo ${cmd}
eval ${cmd}