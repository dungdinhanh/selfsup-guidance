#!/bin/bash


MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 \
 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 \
  --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --fix_seed True"


SAMPLE_FLAGS="--batch_size 1 --num_samples 1 --timestep_respacing 250 --save_imgs_for_visualization True --fix_class True \
  --seed 200"
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

#scales=( "200.0" "0.0")
scales=( "265.0" "267.0"  "280.0" "290.0"  "300.0" "310.0" "340.0" "360.0" "370.0"  "400.0"  )
#scales=( "265.0" "267.0" "268.0"  "275.0" "280.0" "290.0" "295.0" "300.0" "310.0" "320.0" "340.0" "360.0" "370.0" "380.0" "400.0"  )
#scales=( )
#scales=( )
#scales=( )
#scales=( "350.0" "355.0" "357.0" "363.0" )
jointtemps=( "1.0" )
kcs=( "20" )
kchosens=(  "6" "8" "10" "11" "15" "17")
#kchosens=(   "17")
fixclasses=("288")


for scale in "${scales[@]}"
do
for jt in "${jointtemps[@]}"
do
  for kc in "${kcs[@]}"
  do
    for kchosen in "${kchosens[@]}"
    do
      for fixclass in "${fixclasses[@]}"
      do
cmd="python script_odiff/analysis/consistency_analyse.py $MODEL_FLAGS --classifier_scale ${scale}  \
 --classifier_type mocov2 --model_path models/256x256_diffusion.pt $SAMPLE_FLAGS --joint_temperature ${jt}\
 --logdir runs/analysis/consitency/c${fixclass}_k${kchosen}/scale${scale}/ --features eval_models/imn256_mocov2/reps3.npz --k_closest ${kc} \
 --fix_class_index ${fixclass} --k_chosen ${kchosen}"
echo ${cmd}
eval ${cmd}
done
done
done
done
done




#cmd="python scripts_hfai_gdiff/classifier_sample.py --logdir runs/sampling/IMN64/conditional/ \
# ${MODEL_FLAGS} --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt \
# --classifier_depth 4 --model_path models/64x64_diffusion.pt ${SAMPLE_FLAGS}"
#echo ${cmd}
#eval ${cmd}