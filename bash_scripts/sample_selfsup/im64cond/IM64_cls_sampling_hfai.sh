#!/bin/bash

SAMPLE_FLAGS="--batch_size 128 --num_samples 50000 --timestep_respacing 250"
SAMPLE_FLAGS="--batch_size 32 --num_samples 50000 --timestep_respacing 250"
#TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --schedule_sampler loss-second-moment"

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 \
 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3\
 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"


#MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 \
# --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 \
#  --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
cmd="cd ../../../"
echo ${cmd}
eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}

scales=("0.0" "2.0" "4.0")

for scale in "${scales[@]}"
do
cmd="python scripts_gdiff/classifier_sample.py $MODEL_FLAGS --classifier_scale ${scale}  \
--classifier_path models/64x64_classifier.pt --model_path models/64x64_diffusion.pt $SAMPLE_FLAGS \
 --logdir runs/sampling2/IMN64/conditional/scale${scale}/ --classifier_depth 4"
echo ${cmd}
eval ${cmd}
done

for scale in "${scales[@]}"
do
cmd="python evaluations/evaluator_tolog.py reference/VIRTUAL_imagenet64_labeled.npz \
 runs/sampling/IMN64/conditional_mlt/scale${scale}/reference/samples_50000x64x64x3.npz"
#echo ${cmd}
#eval ${cmd}
done




#cmd="python scripts_hfai_gdiff/classifier_sample.py --logdir runs/sampling/IMN64/conditional/ \
# ${MODEL_FLAGS} --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt \
# --classifier_depth 4 --model_path models/64x64_diffusion.pt ${SAMPLE_FLAGS}"
#echo ${cmd}
#eval ${cmd}