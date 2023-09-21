#!/bin/bash


hfai bash bash_scripts/sample_selfsup/im64cond/IM64_cond_ss_sampling_hfai_negw_maxp.sh ++


HFAI_DATASETS_DIR=/ssd/dungda/data/imagenet/ hfai bash bash_scripts/train_sscls/simsiam_negative_ft_step_control/IM256_ss_pretrained_simsiam_maxp700_wneg0p2_ft_300k.sh ++