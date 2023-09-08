#!/bin/bash


iter="500000"
imgs="256"

TRAIN_FLAGS="--iterations ${iter} --anneal_lr True --batch_size 4 --lr 6e-4 --save_interval 10000 --weight_decay 0.2 \
--pretrained_cls simsiam"
CLASSIFIER_FLAGS="--image_size ${imgs} --dim 2048 --pred_dim 512"

cmd="cd ../../../"
echo ${cmd}
eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}

nws=("0.5")

for nw in "${nws[@]}"
do
cmd="python scripts_gdiff/selfsup/classifier_train_selfsup_simsiam_negativew_ft.py --data_dir path/to/imagenet --logdir \
runs/selfsup_training_distanceaware_noT/psimsiam${iter}_IM${imgs}_nw${nw} $TRAIN_FLAGS $CLASSIFIER_FLAGS  --neg_weight ${nw}"
echo ${cmd}
eval ${cmd}
done