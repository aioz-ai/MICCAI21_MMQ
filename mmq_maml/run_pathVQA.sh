#!/usr/bin/env bash
# patHVQA dataset
#  --task_num 4 --n_way 5 --k_spt 5 --k_qry 15
# 5 refinement times
python3 pathVQA_maml_train.py --t_dst 0

python3 pathVQA_maml_half.py --t_dst 1
python3 pathVQA_maml_train.py --t_dst 1

python3 pathVQA_maml_half.py --t_dst 2
python3 pathVQA_maml_train.py --t_dst 2

python3 pathVQA_maml_half.py --t_dst 3
python3 pathVQA_maml_train.py --t_dst 3

python3 pathVQA_maml_half.py --t_dst 4
python3 pathVQA_maml_train.py --t_dst 4

python3 pathVQA_maml_half.py --t_dst 5
python3 pathVQA_maml_train.py --t_dst 5

python3 pathVQA_maml_fuse.py