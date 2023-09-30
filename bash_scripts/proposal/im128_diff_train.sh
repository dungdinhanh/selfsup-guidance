#!/bin/bash

export NCCL_P2P_DISABLE=1

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --image_size 128 \
 --learn_sigma True  --num_channels 256  --num_res_blocks 3 \
  --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 8 --lr_anneal_steps 1 --dropout 0.0 --attention_resolutions 32,16,8 \
 --num_heads 4 --save_interval 50000 "
#total batch size = 16 * 8 * 8 = 1024
cmd="cd ../../"
#echo ${cmd}
#eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}


cmd="python scripts_gdiff/image_train_test.py --data_dir path/to/imagenet --logdir runs/IM128/IM128_diffusion_unconditional/ \
 $TRAIN_FLAGS $DIFFUSION_FLAGS $MODEL_FLAGS"
echo ${cmd}
eval ${cmd}