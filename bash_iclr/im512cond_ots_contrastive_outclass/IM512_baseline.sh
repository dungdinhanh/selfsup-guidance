#!/bin/bash

export NCCL_P2P_DISABLE=1
# MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 \
#  --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 \
#   --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 \
 --image_size 512 --learn_sigma True --noise_schedule linear --num_channels 256 \
 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"

SAMPLE_FLAGS="--batch_size 8 --num_samples 10000 --timestep_respacing 250"



#MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 \
# --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 \
#  --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
# cmd="cd ../../../"
# echo ${cmd}
# eval ${cmd}

base_folder="/hdd/dungda/selfsup-guidance"

cmd="ls"
echo ${cmd}
eval ${cmd}

scales=("4.0")


for scale in "${scales[@]}"
do

cmd="python scripts_gdiff/noclassifier_sample.py $MODEL_FLAGS --classifier_scale ${scale}  \
 --model_path models/512x512_diffusion.pt $SAMPLE_FLAGS \
 --logdir runs/sampling_ots/IMN512_baseline/conditional/scale${scale}/ --base_folder ${base_folder}"
echo ${cmd}
eval ${cmd}
done


for scale in "${scales[@]}"
do

cmd="python evaluations/evaluator_tolog.py ${base_folder}/reference/VIRTUAL_imagenet512.npz \
 ${base_folder}/runs/sampling_ots/IMN512_baseline/conditional/scale${scale}/reference/samples_10000x512x512x3.npz"
echo ${cmd}
eval ${cmd}
done





#cmd="python scripts_hfai_gdiff/classifier_sample.py --logdir runs/sampling/IMN64/conditional/ \
# ${MODEL_FLAGS} --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt \
# --classifier_depth 4 --model_path models/64x64_diffusion.pt ${SAMPLE_FLAGS}"
#echo ${cmd}
#eval ${cmd}