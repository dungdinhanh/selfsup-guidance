#!/bin/bash

#export NCCL_P2P_DISABLE=1

TRAIN_FLAGS="--iterations 10 --anneal_lr True --batch_size 1 --lr 6e-4 --save_interval 10000 --weight_decay 0.2"
CLASSIFIER_FLAGS="--image_size 64 --classifier_attention_resolutions 32,16,8 --classifier_depth 4 \
--classifier_width 128 --classifier_pool attention --classifier_resblock_updown True \
 --classifier_use_scale_shift_norm True --schedule_sampler uniform-with-fix --resume_checkpoint ss_models/scratch_ss_cls.pt \
 --min 980 --max 999"

cmd="cd .."
#echo ${cmd}
#eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}


cmd="python scripts_gdiff/selfsup/analyse/model_check_distance.py --data_dir path/to/imagenet --logdir runs/selfsup_training/scratch_test $TRAIN_FLAGS $CLASSIFIER_FLAGS"
echo ${cmd}
eval ${cmd}