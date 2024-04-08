#!/bin/bash

# Loading the required module
source /etc/profile
module load anaconda/2023a-pytorch
ulimit -n 4096


# python get_pairs.py --directory 'native_protein' --output_csv 'pairwise_counts_native_3A.csv' --distance 3


python get_pairs.py --directory 'native_protein' --output_csv 'pairwise_counts_native_5A.csv' --distance 5