#!/bin/bash

export NCCL_P2P_DISABLE=1
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 \
 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 \
  --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

 MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 \
 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3\
 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"


SAMPLE_FLAGS="--batch_size 200 --num_samples 30 --timestep_respacing 250"
#SAMPLE_FLAGS="--batch_size 2 --num_samples 2 --timestep_respacing 250"



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

scales=( "20.0" )
jointtemps=( "2.0" )
kcs=("1" "5" "10" "15" "20")


for scale in "${scales[@]}"
do
for jt in "${jointtemps[@]}"
do
  for kc in "${kcs[@]}"
  do
cmd="python script_odiff/analysis/mocov2_meanclose_contrastive_outclass_sup_instance_sample_transform_time.py $MODEL_FLAGS --classifier_scale ${scale}  \
 --classifier_type mocov2 --model_path models/64x64_diffusion.pt $SAMPLE_FLAGS --joint_temperature ${jt}\
 --logdir runs/sampling_ots_bigk/IMN64/time/kc${kc}/conditional/scale${scale}_mocov2_mean_close/ --features eval_models/imn64_mocov2/reps3.npz --k_closest  ${kc} --base_folder ${base_folder}"
echo ${cmd}
eval ${cmd}
done
done
done





#cmd="python scripts_hfai_gdiff/classifier_sample.py --logdir runs/sampling/IMN64/conditional/ \
# ${MODEL_FLAGS} --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt \
# --classifier_depth 4 --model_path models/64x64_diffusion.pt ${SAMPLE_FLAGS}"
#echo ${cmd}
#eval ${cmd}