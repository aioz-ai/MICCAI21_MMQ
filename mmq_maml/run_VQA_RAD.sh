#!/usr/bin/env bash
# VQA-RAD dataset
#  --task_num 5 --n_way 3 --k_spt 3 --k_qry 3
# 5 refinement times
python3 VQA_RAD_train.py  --t_dst 0

python3 VQA_RAD_half.py --t_dst 1
python3 VQA_RAD_train.py  --t_dst 1

python3 VQA_RAD_half.py --t_dst 2
python3 VQA_RAD_train.py  --t_dst 2

python3 VQA_RAD_half.py --t_dst 3
python3 VQA_RAD_train.py  --t_dst 3

python3 VQA_RAD_half.py --t_dst 4
python3 VQA_RAD_train.py  --t_dst 4

python3 VQA_RAD_half.py --t_dst 5
python3 VQA_RAD_train.py  --t_dst 5

python3 VQA_RAD_fuse.py