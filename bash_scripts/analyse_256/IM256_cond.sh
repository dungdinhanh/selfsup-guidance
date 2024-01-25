#!/bin/bash


MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 \
 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 \
  --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --fix_seed True"


SAMPLE_FLAGS="--batch_size 1 --num_samples 1 --timestep_respacing 250 --save_imgs_for_visualization True --fix_class True \
 --fix_class_index 288 --seed 200"
#SAMPLE_FLAGS="--batch_size 2 --num_samples 2 --timestep_respacing 250"



#MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 \
# --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 \
#  --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
cmd="cd ../../../"
#echo ${cmd}
#eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}

scales=( "21.0" "0.0")
jointtemps=( "1.0" )
kcs=( "20" )


for scale in "${scales[@]}"
do
for jt in "${jointtemps[@]}"
do
  for kc in "${kcs[@]}"
  do
cmd="python script_odiff/analysis/consistency_analyse.py $MODEL_FLAGS --classifier_scale ${scale}  \
 --classifier_type mocov2 --model_path models/256x256_diffusion.pt $SAMPLE_FLAGS --joint_temperature ${jt}\
 --logdir runs/analysis/scale${scale}_analyse_consistency/ --features eval_models/imn256_mocov2/reps3.npz --k_closest 20"
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
cmd="python evaluations/evaluator_tolog.py reference/VIRTUAL_imagenet256_labeled.npz \
 runs/sampling_ots_bigk_contrastive/IMN256/kc${kc}/conditional/scale${scale}_mocov2_mean_close/reference/samples_50000x256x256x3.npz"
#echo ${cmd}
#eval ${cmd}
done
done
done




#cmd="python scripts_hfai_gdiff/classifier_sample.py --logdir runs/sampling/IMN64/conditional/ \
# ${MODEL_FLAGS} --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt \
# --classifier_depth 4 --model_path models/64x64_diffusion.pt ${SAMPLE_FLAGS}"
#echo ${cmd}
#eval ${cmd}