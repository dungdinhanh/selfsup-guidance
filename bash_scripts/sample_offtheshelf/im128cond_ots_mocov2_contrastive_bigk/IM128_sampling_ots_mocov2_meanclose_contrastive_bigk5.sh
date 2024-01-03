#!/bin/bash

export NCCL_P2P_DISABLE=1
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 128 \
--learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True\
 --use_fp16 True --use_scale_shift_norm True"

SAMPLE_FLAGS="--batch_size 100 --num_samples 50000 --timestep_respacing 256"
#SAMPLE_FLAGS="--batch_size 2 --num_samples 2 --timestep_respacing 250"



cmd="cd ../../../"
#echo ${cmd}
#eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}

scales=(  "20.0" )
jointtemps=(  "1.0" )
margintemps=( "2.0" "4.0" "6.0")
kcs=("20")


for scale in "${scales[@]}"
do
for jt in "${jointtemps[@]}"
do
for mt in "${margintemps[@]}"
do
  for kc in "${kcs[@]}"
  do
cmd="python script_odiff/mocov2_meanclose_contrastive_sup_instance_sample_transform.py $MODEL_FLAGS --classifier_scale ${scale} \
 --classifier_type mocov2 \
--model_path models/128x128_diffusion.pt $SAMPLE_FLAGS --joint_temperature ${jt} --margin_temperature_discount ${mt}  \
--logdir runs/sampling_ots_bigk_contrastive/IMN128/kc${kc}/conditional/scale${scale}_jointtemp${jt}_margtemp${mt}_mocov2_meanclose/ \
--features eval_models/imn128_mocov2/reps3.npz --k_closest ${kc}"
echo ${cmd}
eval ${cmd}
done
done
done
done
#
#scales=("5.0" "7.0" )

for scale in "${scales[@]}"
do
for jt in "${jointtemps[@]}"
do
for mt in "${margintemps[@]}"
do
  for kc in "${kcs[@]}"
  do
cmd="python evaluations/evaluator_tolog.py reference/VIRTUAL_imagenet128_labeled.npz \
 runs/sampling_ots_bigk_contrastive/IMN128/kc${kc}/conditional/scale${scale}_jointtemp${jt}_margtemp${mt}_mocov2_meanclose/reference/samples_50000x128x128x3.npz"
echo ${cmd}
eval ${cmd}
done
done
done
done




#cmd="python scripts_hfai_gdiff/classifier_free_sample.py --logdir runs/classifier_pretrained/ ${MODEL_FLAGS} --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt \
# --classifier_depth 4 --model_path models/64x64_diffusion.pt ${SAMPLE_FLAGS}"