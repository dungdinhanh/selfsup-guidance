#!/bin/bash

export NCCL_P2P_DISABLE=1

SAMPLE_FLAGS="--batch_size 220 --num_samples 50000 --timestep_respacing 250"
#SAMPLE_FLAGS="--batch_size 2 --num_samples 4 --timestep_respacing 250"
#SAMPLE_FLAGS="--batch_size 2 --num_samples 4 --timestep_respacing 250"
#SAMPLE_FLAGS="--batch_size 32 --num_samples 50000 --timestep_respacing 250"
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

<<<<<<< HEAD
scales=(  "18.0" "0.001" "0.005")
=======

scales=( "0.001" "0.005")

>>>>>>> c5a612c7292a8da6b53d66b37630b668aa3a4c26
#scales=( "10.0"  )
#scales=( "1.0"  )
jointtemps=( "0.3")
margintemps=( "0.3" )


for scale in "${scales[@]}"
do
for jt in "${jointtemps[@]}"
do
for mt in "${margintemps[@]}"
do
cmd="python script_odiff/mocov2_meanclose_sm_contrastive_sup_instance_sample_transform.py $MODEL_FLAGS --classifier_scale ${scale}  \
--classifier_type mocov2 --model_path models/64x64_diffusion.pt $SAMPLE_FLAGS --joint_temperature ${jt} \
 --logdir runs/sampling_ots/IMN64/conditional/scale${scale}_jointtemp${jt}_margtemp${mt}_mocov2_meanclose_sup_smcontrastive_isb/ \
 --features eval_models/imn64_mocov2/reps3.npz --save_imgs_for_visualization True"
echo ${cmd}
eval ${cmd}
done
done
done

for scale in "${scales[@]}"
do
for jt in "${jointtemps[@]}"
do
for mt in "${margintemps[@]}"
do
cmd="python evaluations/evaluator_tolog.py reference/VIRTUAL_imagenet64_labeled.npz \
 runs/sampling_ots/IMN64/conditional/scale${scale}_jointtemp${jt}_margtemp${mt}_mocov2_meanclose_sup_smcontrastive_isb/reference/samples_50000x64x64x3.npz"
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