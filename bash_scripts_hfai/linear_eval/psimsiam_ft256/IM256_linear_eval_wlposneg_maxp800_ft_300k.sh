#!/bin/bash

<<<<<<< HEAD
#export NCCL_P2P_DISABLE=1
=======
export NCCL_P2P_DISABLE=1
>>>>>>> f1ed5f0ea6ff3e99b3ef6fcf6642ea7b988ca461
#iter="1000000"
#imgs="128"

#TRAIN_FLAGS="--iterations ${iter} --anneal_lr True --batch_size 8 --lr 6e-4 --save_interval 10000 --weight_decay 0.2 \
#--pretrained_cls simsiam"
#CLASSIFIER_FLAGS="--image_size ${imgs} --dim 2048 --pred_dim 512"

cmd="cd ../../../"
<<<<<<< HEAD
echo ${cmd}
eval ${cmd}
=======
#echo ${cmd}
#eval ${cmd}
>>>>>>> f1ed5f0ea6ff3e99b3ef6fcf6642ea7b988ca461

cmd="ls"
echo ${cmd}
eval ${cmd}


# shellcheck disable=SC2089
cmd="python scripts_gdiff/selfsup/analyse/simsiam/main_lincls_normdiff_pdiff.py\
  -a resnet50 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed \
<<<<<<< HEAD
  --world-size 1 --rank 0  --pretrained runs/selfsup_training_distanceaware_noT/psimsiam300000_IM256_wneg0.5_maxp700/models/model070000.pt  \
  --lars --image_size 256 --save_folder runs/linear_eval/im256_lin_wl_maxp700"
=======
  --world-size 1 --rank 0  --pretrained runs/selfsup_training_distanceaware_noT/psimsiam300000_IM256_wlposneg_maxp800/models/model090000.pt  \
  --lars --image_size 256 --save_folder runs/linear_eval/im256_lin_wlposneg_maxp800"
>>>>>>> f1ed5f0ea6ff3e99b3ef6fcf6642ea7b988ca461
#   --world-size 1 --rank 0  --pretrained runs/selfsup_training_distanceaware_noT/psimsiam150000/models/model149999.pt --lars"
echo ${cmd}
eval ${cmd}