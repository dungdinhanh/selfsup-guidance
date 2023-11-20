#!/bin/bash


MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 \
 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 \
  --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"


SAMPLE_FLAGS="--batch_size 50 --num_samples 50000 --timestep_respacing 250"



#MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 \
# --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 \
#  --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
cmd="cd ../../../"
echo ${cmd}
eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}

scales=("8.0" "10.0")
jointtemps=("1.0" "0.5" "0.3")


for scale in "${scales[@]}"
do
for jt in "${jointtemps[@]}"
do
cmd="python script_odiff/classifier_sample.py $MODEL_FLAGS --classifier_scale ${scale}  \
 --classifier_type resnet50 --model_path models/256x256_diffusion_uncond.pt $SAMPLE_FLAGS --joint_temperature ${jt} \
 --logdir runs/sampling_ots/IMN256/unconditional/scale${scale}/scale${scale}_jointtemp${jt}/ "
echo ${cmd}
eval ${cmd}
done
done

for scale in "${scales[@]}"
do
for jt in "${jointtemps[@]}"
do
cmd="python evaluations/evaluator_tolog.py reference/VIRTUAL_imagenet256_labeled.npz \
 runs/sampling_ots/IMN256/unconditional/scale${scale}/scale${scale}_jointtemp${jt}/reference/samples_50000x256x256x3.npz"
echo ${cmd}
eval ${cmd}
done
done




#cmd="python scripts_hfai_gdiff/classifier_sample.py --logdir runs/sampling/IMN64/conditional/ \
# ${MODEL_FLAGS} --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt \
# --classifier_depth 4 --model_path models/64x64_diffusion.pt ${SAMPLE_FLAGS}"
#echo ${cmd}
#eval ${cmd}