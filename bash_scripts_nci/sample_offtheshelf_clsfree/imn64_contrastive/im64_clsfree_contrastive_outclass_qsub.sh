#!/bin/bash

cscales=("10.0" "12.0" "14.0" "16.0" "18.0")

for cscale in "${cscales[@]}"
do
  cmd="qsub -v cscale='${cscale}' bash_scripts_nci/sample_offtheshelf_clsfree/imn64_contrastive/im64_clsfree_contrastive_outclass_tunec.sh \
   -o output_nci/log_clsfree_contrastive_tunec${cscale}.txt  -e output_nci/error_clsfree_contrastive_tunec${cscale}.txt "
   echo ${cmd}
   eval ${cmd}
   done
#PBS -o output_nci/log5.txt
#PBS -e output_nci/error5.txt