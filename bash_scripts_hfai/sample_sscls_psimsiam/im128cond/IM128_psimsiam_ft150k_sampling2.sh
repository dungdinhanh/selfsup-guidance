#!/bin/bash


MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 128 \
--learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True\
 --use_fp16 True --use_scale_shift_norm True"

SAMPLE_FLAGS="--batch_size 80 --num_samples 50000 --timestep_respacing 250"



cmd="cd ../../../"
echo ${cmd}
eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}

scales=("3.0" "4.0" "5.0" "6.0")
fts=("150000")


for scale in "${scales[@]}"
do
  for ft in "${fts[@]}"
  do
cmd="python scripts_gdiff/selfsup/classifier_sample_ss_psimsiam.py $MODEL_FLAGS --classifier_scale ${scale} \
 --classifier_path runs/selfsup_training_distanceaware_noT/psimsiam${ft}_IM128/models/model149999.pt \
--model_path models/128x128_diffusion.pt $SAMPLE_FLAGS  --logdir runs/ft_negweight/sampling/IMN128_sft${ft}/conditional/scale${scale}/"
echo ${cmd}
eval ${cmd}
done
done


for scale in "${scales[@]}"
do
for ft in "${fts[@]}"
do
cmd="python evaluations/evaluator_tolog.py reference/VIRTUAL_imagenet128_labeled.npz \
 runs/ft_negweight/sampling/IMN128_sft${ft}/conditional/scale${scale}/reference/samples_50000x128x128x3.npz"
echo ${cmd}
eval ${cmd}
done
done




#cmd="python scripts_hfai_gdiff/classifier_free_sample.py --logdir runs/classifier_pretrained/ ${MODEL_FLAGS} --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt \
# --classifier_depth 4 --model_path models/64x64_diffusion.pt ${SAMPLE_FLAGS}"