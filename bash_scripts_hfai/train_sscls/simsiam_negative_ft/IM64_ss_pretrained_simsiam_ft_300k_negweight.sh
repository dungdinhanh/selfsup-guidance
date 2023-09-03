#!/bin/bash


iter="300000"

TRAIN_FLAGS="--iterations ${iter} --anneal_lr True --batch_size 8 --lr 6e-4 --save_interval 10000 --weight_decay 0.2 \
--pretrained_cls simsiam "
CLASSIFIER_FLAGS="--image_size 64 --dim 2048 --pred_dim 512"

cmd="cd ../../../"
echo ${cmd}
eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}

nws=("1.0" "2.0" "2.5" "4.0" )
#scales=( "6.0" "7.0" "8.0" "9.0" "10.0")
#
#
#for scale in "${scales[@]}"
#do

for nw in "${nws[@]}"
do
cmd="python scripts_gdiff/selfsup/classifier_train_selfsup_simsiam_negativew_ft.py --data_dir path/to/imagenet --logdir \
runs/selfsup_training_distanceaware_noT/IMN64/psimsiam${iter}_nw${nw} $TRAIN_FLAGS $CLASSIFIER_FLAGS --neg_weight ${nw}"
echo ${cmd}
eval ${cmd}
done