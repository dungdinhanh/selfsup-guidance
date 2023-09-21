#!/bin/bash

#export NCCL_P2P_DISABLE=1
iter="300000"
imgs="256"

TRAIN_FLAGS="--iterations ${iter} --anneal_lr True --batch_size 8 --lr 6e-4 --save_interval 10000 --weight_decay 0.2 \
--pretrained_cls simsiam"
CLASSIFIER_FLAGS="--image_size ${imgs} --dim 2048 --pred_dim 512"

cmd="cd ../../../"
echo ${cmd}
eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}


cmd="python scripts_gdiff/selfsup/classifier_train_selfsup_simsiam_samplercontrol_maxp_wnegative_ft.py --data_dir path/to/imagenet --logdir \
runs/selfsup_training_distanceaware_noT/psimsiam${iter}_IM${imgs}_wneg0.0_maxp400 --wneg 0.0 $TRAIN_FLAGS $CLASSIFIER_FLAGS --maxtime 400"
echo ${cmd}
eval ${cmd}