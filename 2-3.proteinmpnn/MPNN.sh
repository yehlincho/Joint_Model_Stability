#!/bin/bash

# Loading the required module
source /etc/profile
module load anaconda/2023a-pytorch
ulimit -n 4096


# python ProteinMPNN.py --data_dir '/home/gridsan/ylcho/DMSV2/dataset/af2_predicted' --save_dir '/home/gridsan/ylcho/DMSV2/conditional_MPNN_pssm_v_48_010.txt' --model_name 'v_48_010'

python ProteinMPNN.py --data_dir '/home/gridsan/ylcho/DMSV2/dataset/af2_predicted' --save_dir '/home/gridsan/ylcho/DMSV2/unconditional_MPNN_pssm_v_48_010.txt' --model_name 'v_48_010'  --unconditional_logits True
