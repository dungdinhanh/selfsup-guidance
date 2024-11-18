#!/bin/bash

##hfai bash bash_scripts_hfai/sample_consistency/im64unc_consist/IM64_unc_sampling_cons.sh ++
#
#hfai bash bash_scripts_hfai/sample_consistency/im128cond_consist/IM128_sampling_consist_kd.sh ++
#
##hfai bash bash_scripts_hfai/sample_consistency/im64unc_consist/IM64_unc_sampling_cons2.sh ++

hfai workspace push --force --oss_timeout 43200 --list_timeout 7200 --token_expires 43200 --sync_timeout 21600

hfai workspace push --force --oss_timeout 43200 --list_timeout 7200 --token_expires 43200 --sync_timeout 21600

HF_ENV_NAME=dbgcxe4 hfai bash bash_scripts_hfai/sample_consistency/im256_lsun_bed_consist/IM256_lsun_clseds_sampling_hfai.sh -- -n 50 --name im256_lsun_bed_kd1 --force

HF_ENV_NAME=dbgcxe4 hfai bash bash_scripts_hfai/sample_consistency/im256_lsun_bed_consist/IM256_lsun_clseds_sampling_hfai2.sh -- -n 50 --name im256_lsun_bed_kd2 --force

HF_ENV_NAME=dbgcxe4 hfai bash bash_scripts_hfai/sample_consistency/im256_lsun_bed_consist/IM256_lsun_clseds_sampling_hfai3.sh -- -n 50 --name im256_lsun_bed_kd3 --force