#!/bin/bash

#PBS -q gpuvolta
#PBS -P jp09
#PBS -l walltime=48:00:00
#PBS -l mem=256GB
#PBS -l ncpus=48
#PBS -l ngpus=4
#PBS -l jobfs=256GB
#PBS -l wd
#PBS -l storage=scratch/jp09
#PBS -M adin6536@uni.sydney.edu.au

module load use.own
module load python3/3.9.2
module load sdiff

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 \
 --image_size 512 --learn_sigma True --noise_schedule linear --num_channels 256 \
 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"


SAMPLE_FLAGS="--batch_size 14 --num_samples 10000 --timestep_respacing 250"
# SAMPLE_FLAGS="--batch_size 2 --num_samples 4 --timestep_respacing 250"



#MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 \
# --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 \
#  --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
# cmd="cd ../../../"
# echo ${cmd}
# eval ${cmd}

base_folder="./"

cmd="ls"
echo ${cmd}
eval ${cmd}

scales=( "20.0" )
jointtemps=(  "2.0")
kcs=( "20")



for scale in "${scales[@]}"
do
for jt in "${jointtemps[@]}"
do
  for kc in "${kcs[@]}"
  do
cmd="WORLD_SIZE=1 RANK=0 MASTER_IP=127.0.0.1 MASTER_PORT=29510 MARSV2_WHOLE_LIFE_STATE=0 python3 script_odiff/mocov2_meanclose_contrastive_outclass_sup_instance_sample_transform.py $MODEL_FLAGS --classifier_scale ${scale}  \
 --classifier_type mocov2 --model_path models/512x512_diffusion.pt $SAMPLE_FLAGS --joint_temperature ${jt}\
 --logdir runs/sampling_ots_contrastive_outclass2/IMN512/kc${kc}/conditional/scale${scale}_jt${jt}_mocov2_mean_close/ --features eval_models/imn512_mocov2/reps3.npz --k_closest 10"
echo ${cmd}
eval ${cmd}
done
done
done

for scale in "${scales[@]}"
do
for jt in "${jointtemps[@]}"
do
  for kc in "${kcs[@]}"
  do
cmd="python evaluations/evaluator_tolog.py ${base_folder}/reference/VIRTUAL_imagenet512.npz \
 ${base_folder}/runs/sampling_ots_contrastive_outclass2/IMN512/kc${kc}/conditional/scale${scale}_jt${jt}_mocov2_mean_close/reference/samples_10000x512x512x3.npz"
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