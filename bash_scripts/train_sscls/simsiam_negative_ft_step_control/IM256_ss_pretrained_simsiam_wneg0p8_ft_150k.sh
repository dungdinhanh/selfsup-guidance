#!/bin/bash

export NCCL_P2P_DISABLE=1
iter="150000"
imgs="256"

TRAIN_FLAGS="--iterations ${iter} --anneal_lr True --batch_size 64 --lr 6e-4 --save_interval 10000 --weight_decay 0.2 \
--pretrained_cls simsiam"
CLASSIFIER_FLAGS="--image_size ${imgs} --dim 2048 --pred_dim 512"

cmd="cd ../../../"
#echo ${cmd}
#eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}


cmd="python scripts_gdiff/selfsup/classifier_train_selfsup_simsiam_samplercontrol_wnegative_ft.py --data_dir path/to/imagenet --logdir \
runs/selfsup_training_distanceaware_samplercontroller/psimsiam${iter}_IM${imgs}_wneg0.8 --wneg 0.8 $TRAIN_FLAGS $CLASSIFIER_FLAGS"
echo ${cmd}
eval ${cmd}