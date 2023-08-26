#!/bin/bash

export NCCL_P2P_DISABLE=1

iter="50000"

TRAIN_FLAGS="--iterations ${iter} --anneal_lr True --batch_size 60 --lr 6e-4 --save_interval 10000 --weight_decay 0.2 \
--pretrained_cls models/64x64_classifier.pt"
CLASSIFIER_FLAGS="--image_size 64 --dim 2048 --pred_dim 512"

cmd="cd .."
#echo ${cmd}
#eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}


cmd="python scripts_gdiff/selfsup/classifier_train_selfsup_zerodistance_not_simsiam.py --data_dir path/to/imagenet --logdir \
runs/selfsup_training_distanceaware_noT_0distance/pretrainedcls${iter} $TRAIN_FLAGS $CLASSIFIER_FLAGS"
echo ${cmd}
eval ${cmd}