#!/bin/bash

export NCCL_P2P_DISABLE=1
#iter="1000000"
#imgs="128"

#TRAIN_FLAGS="--iterations ${iter} --anneal_lr True --batch_size 8 --lr 6e-4 --save_interval 10000 --weight_decay 0.2 \
#--pretrained_cls simsiam"
#CLASSIFIER_FLAGS="--image_size ${imgs} --dim 2048 --pred_dim 512"

cmd="cd ../../../"
#echo ${cmd}
#eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}


# shellcheck disable=SC2089
cmd="python scripts_gdiff/selfsup/analyse/simsiam/main_lincls_normdiff_pdiff_diffaug.py \
  -a resnet50 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed \
  --world-size 1 --rank 0  --pretrained runs/selfsup_training_distanceaware_noT/psimsiam150000/models/model149999.pt  \
  --lars --image_size 64 --save_folder runs/linear_eval_noisy/im64_lin_negw"
#   --world-size 1 --rank 0  --pretrained runs/selfsup_training_distanceaware_noT/psimsiam150000/models/model149999.pt --lars"
echo ${cmd}
eval ${cmd}