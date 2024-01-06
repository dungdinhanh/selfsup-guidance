#!/bin/bash

export NCCL_P2P_DISABLE=1

SAMPLE_FLAGS="--batch_size 30 --num_samples 50000 --timestep_respacing 250"
#SAMPLE_FLAGS="--batch_size 4 --num_samples 50000 --timestep_respacing 250"
#TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --schedule_sampler loss-second-moment"

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 \
 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 \
  --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"


#MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 \
# --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 \
#  --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
cmd="cd ../../../"
#echo ${cmd}
#eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}

scales=(  "0.1" "0.5" "1.0" "2.0")
scales=(  "0.1" )

for scale in "${scales[@]}"
do
cmd="python script_odiff/classifier_free/classifier_free_sample2.py $MODEL_FLAGS --cond_model_scale ${scale}  \
--uncond_model_path models/256x256_diffusion_uncond.pt \
--model_path models/256x256_diffusion.pt   $SAMPLE_FLAGS \
 --logdir runs/sampling_clsfree_version2/IMN256/normal/scale${scale}/ "
echo ${cmd}
eval ${cmd}
done



for scale in "${scales[@]}"
do
cmd="python evaluations/evaluator_tolog.py reference/VIRTUAL_imagenet256_labeled.npz \
 runs/sampling_clsfree_version2/IMN256/normal/scale${scale}/reference/samples_50000x64x64x3.npz"
#echo ${cmd}
#eval ${cmd}
done
