#!/bin/bash

export NCCL_P2P_DISABLE=1

iter="20"

TRAIN_FLAGS="--iterations ${iter} --anneal_lr True --batch_size 1 --lr 6e-4 --save_interval 1 --weight_decay 0.2 \
--resume_checkpoint runs/selfsup_training_distanceaware_noT/psimsiam150000/models/model149999.pt"
CLASSIFIER_FLAGS="--dim 2048 --pred_dim 512"

cmd="cd .."
#echo ${cmd}
#eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}


cmd="python scripts_gdiff/selfsup/analyse/simsiam_loadsimsiam_check_distance_2image_t0.py --data_dir path/to/imagenet \
 --logdir runs/selfsup_training_distanceaware/pretrainedcls${iter}/test/ \
 $TRAIN_FLAGS $CLASSIFIER_FLAGS"
echo ${cmd}
eval ${cmd}