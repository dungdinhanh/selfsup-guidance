#!/bin/bash

TRAIN_FLAGS="--iterations 300000 --anneal_lr True --batch_size 10 --lr 6e-4 --save_interval 10000 --weight_decay 0.2"
CLASSIFIER_FLAGS="--image_size 256 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 \
--classifier_width 128 --classifier_pool attention --classifier_resblock_updown True \
 --classifier_use_scale_shift_norm True"


cmd="cd ../../"
echo ${cmd}
eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}


cmd="CUDA_VISIBLE_DEVICES=0 python scripts_gdiff/consistency/classifier_gather_rep3.py --data_dir path/to/imagenet \
--logdir eval_models/imn256 --p_classifier models/256x256_classifier.pt \
 $TRAIN_FLAGS $CLASSIFIER_FLAGS"
echo ${cmd}
eval ${cmd}