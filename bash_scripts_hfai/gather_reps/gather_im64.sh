#!/bin/bash

TRAIN_FLAGS="--iterations 300000 --anneal_lr True --batch_size 32 --lr 6e-4 --save_interval 10000 --weight_decay 0.2"
CLASSIFIER_FLAGS="--image_size 64 --classifier_attention_resolutions 32,16,8 --classifier_depth 4 \
--classifier_width 128 --classifier_pool attention --classifier_resblock_updown True \
 --classifier_use_scale_shift_norm True"

cmd="cd .."
#echo ${cmd}
#eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}


cmd="CUDA_VISIBLE_DEVICES=3 MASTER_PORT=6532 python scripts_gdiff/consistency/classifier_gather_rep.py --data_dir path/to/imagenet --logdir runs/IM64/IM64classifier_training_info_4nodes/ \
 $TRAIN_FLAGS $CLASSIFIER_FLAGS"
echo ${cmd}
eval ${cmd}