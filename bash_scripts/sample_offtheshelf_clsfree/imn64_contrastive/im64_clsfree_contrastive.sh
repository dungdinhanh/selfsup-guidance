#!/bin/bash

export NCCL_P2P_DISABLE=1

SAMPLE_FLAGS="--batch_size 200 --num_samples 50000 --timestep_respacing 250"
#SAMPLE_FLAGS="--batch_size 1 --num_samples 1 --timestep_respacing 250"
#SAMPLE_FLAGS="--batch_size 4 --num_samples 50000 --timestep_respacing 250"
#TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --schedule_sampler loss-second-moment"

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 \
 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3\
 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"


#MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 \
# --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 \
#  --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
cmd="cd ../../../"
#echo ${cmd}
#eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}

#scales=( "0.05" "0.1" "0.2")
scales=( "0.1")
cscales=("10.0" "12.0" "14.0" "16.0" "18.0")
cscales=("10.0" "12.0" "14.0" "16.0" )
jointtemps=("1.0")
margintemps=("1.0")
kcs=("5")

for scale in "${scales[@]}"
do
  for cscale in "${cscales[@]}"
  do
    for jt in "${jointtemps[@]}"
do
for mt in "${margintemps[@]}"
do
  for kc in "${kcs[@]}"
  do
cmd="python script_odiff/classifier_free/classifier_free_sample2_contrastive.py $MODEL_FLAGS --cond_model_scale ${scale}  \
--uncond_model_path models/64x64_diffusion_unc.pt --classifier_type mocov2 \
 --features eval_models/imn64_mocov2/reps3.npz --classifier_scale ${cscale}\
 --save_imgs_for_visualization True \
--model_path models/64x64_diffusion.pt  $SAMPLE_FLAGS \
 --k_closest ${kc} --joint_temperature ${jt} --margin_temperature_discount ${mt}\
 --logdir runs/sampling_clsfree_version2/IMN64/contrastive/scale${scale}_cscale${cscale}_jt${jt}_mt${mt}/ "
echo ${cmd}
eval ${cmd}
done
done
done
done
done


for scale in "${scales[@]}"
do
  for cscale in "${cscales[@]}"
  do
    for jt in "${jointtemps[@]}"
do
for mt in "${margintemps[@]}"
do
cmd="python evaluations/evaluator_tolog.py reference/VIRTUAL_imagenet64_labeled.npz \
 runs/sampling_clsfree_version2/IMN64/contrastive/scale${scale}_cscale${cscale}_jt${jt}_mt${mt}/reference/samples_50000x64x64x3.npz"
echo ${cmd}
eval ${cmd}
done
done
done
done