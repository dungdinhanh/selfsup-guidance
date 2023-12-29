#!/bin/bash


export NCCL_P2P_DISABLE=1

MODEL_FLAGS=""

SAMPLE_FLAGS="--batch_size 70 --num_samples 30000 --timestep_respacing 250"
#SAMPLE_FLAGS="--batch_size 2 --num_samples 6 --timestep_respacing 250"
#export NCCL_P2P_DISABLE=1

cmd="cd ../../../"
#echo ${cmd}
#eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}

#scales=( "2.0" "2.5" "3.0"  )
scales=( "1.0"  )

epss=("0.8" "0.9" "0.95")

ext_capts=("1000")


for scale in "${scales[@]}"
do
  for eps in "${epss[@]}"
  do
    for extcapt in "${ext_capts[@]}"
    do
cmd="python scripts_glide/contrastive/glide_up_sample_contrastive.py $MODEL_FLAGS --guidance_scale ${scale} \
 --ext_captions eval_models/pretext2img/reference/captions_${extcapt}_512.npz --eps ${eps}  $SAMPLE_FLAGS \
 --logdir runs/sampling_glide_contrastive/IMN256/scale${scale}_eps${eps}_ec${extcapt}/ "
echo ${cmd}
eval ${cmd}
done
done
done

#for scale in "${scales[@]}"
#do
#cmd="python evaluations/evaluator_tolog.py reference/VIRTUAL_MSCOCO_val_256x256.npz \
# runs/sampling_glide/IMN256/scale${scale}/reference/samples_30000x256x256x3.npz"
#echo ${cmd}
#eval ${cmd}
#done
for scale in "${scales[@]}"
do
  for eps in "${epss[@]}"
  do
    for extcapt in "${ext_capts[@]}"
    do
cmd="python evaluations/evaluator_tolog.py reference/VIRTUAL_MSCOCO_val_64x64_squ.npz \
 runs/sampling_glide_contrastive/IMN256/scale${scale}_eps${eps}_ec${extcapt}/reference/samples_30000x64x64x3.npz"
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