#!/usr/bin/env bash
#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --output saved_models/rad_BAN_maml_84_pytorch_t1_other_3shot --maml_nums 1 --model BAN --feat_dim 64
#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --output saved_models/rad_BAN_maml_84_pytorch_t2_other_3shot --maml_nums 2 --model BAN --feat_dim 64
#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --output saved_models/rad_BAN_maml_84_pytorch_t3_other_3shot --maml_nums 3 --model BAN --feat_dim 64
#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --output saved_models/rad_BAN_maml_84_pytorch_t4_other_3shot --maml_nums 4 --model BAN --feat_dim 64
#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --output saved_models/rad_BAN_maml_84_pytorch_t5_other_3shot --maml_nums 5 --model BAN --feat_dim 64
#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --output saved_models/rad_BAN_maml_84_pytorch_t-1_other_3shot --maml_nums -1 --model BAN --feat_dim 64

#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --input saved_models/rad_BAN_maml_84_pytorch_t1_other_3shot --epoch 39 --maml_nums 1 --model BAN --feat_dim 64
#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --input saved_models/rad_BAN_maml_84_pytorch_t2_other_3shot --epoch 39 --maml_nums 2 --model BAN --feat_dim 64
#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --input saved_models/rad_BAN_maml_84_pytorch_t3_other_3shot --epoch 39 --maml_nums 3 --model BAN --feat_dim 64
#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --input saved_models/rad_BAN_maml_84_pytorch_t4_other_3shot --epoch 39 --maml_nums 4 --model BAN --feat_dim 64
#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --input saved_models/rad_BAN_maml_84_pytorch_t5_other_3shot --epoch 39 --maml_nums 5 --model BAN --feat_dim 64
#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --input saved_models/rad_BAN_maml_84_pytorch_t-1_other_3shot --epoch 39 --maml_nums -1 --model BAN --feat_dim 64

#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --output saved_models/rad_BAN_maml_84_pytorch_t1_other_3shot_newmethod --maml_nums 1 --model BAN --feat_dim 64
#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --output saved_models/rad_BAN_maml_84_pytorch_t2_other_3shot_newmethod --maml_nums 2 --model BAN --feat_dim 64
#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --output saved_models/rad_BAN_maml_84_pytorch_t3_other_3shot_newmethod --maml_nums 3 --model BAN --feat_dim 64
#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --output saved_models/rad_BAN_maml_84_pytorch_t4_other_3shot_newmethod --maml_nums 4 --model BAN --feat_dim 64
#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --output saved_models/rad_BAN_maml_84_pytorch_t5_other_3shot_newmethod --maml_nums 5 --model BAN --feat_dim 64
#
#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --output saved_models/rad_BAN_maml_dae_84_pytorch_t1_other_3shot_newmethod --maml_nums 1 --model BAN --feat_dim 64 --autoencoder
#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --output saved_models/rad_BAN_maml_dae_84_pytorch_t2_other_3shot_newmethod --maml_nums 2 --model BAN --feat_dim 64 --autoencoder
#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --output saved_models/rad_BAN_maml_dae_84_pytorch_t3_other_3shot_newmethod --maml_nums 3 --model BAN --feat_dim 64 --autoencoder
#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --output saved_models/rad_BAN_maml_dae_84_pytorch_t4_other_3shot_newmethod --maml_nums 4 --model BAN --feat_dim 64 --autoencoder
#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --output saved_models/rad_BAN_maml_dae_84_pytorch_t5_other_3shot_newmethod --maml_nums 5 --model BAN --feat_dim 64 --autoencoder

#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --input saved_models/rad_BAN_maml_84_pytorch_t1_other_3shot_newmethod --epoch 39 --maml_nums 1 --model BAN --feat_dim 64
#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --input saved_models/rad_BAN_maml_84_pytorch_t2_other_3shot_newmethod --epoch 39 --maml_nums 2 --model BAN --feat_dim 64
#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --input saved_models/rad_BAN_maml_84_pytorch_t3_other_3shot_newmethod --epoch 39 --maml_nums 3 --model BAN --feat_dim 64
#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --input saved_models/rad_BAN_maml_84_pytorch_t4_other_3shot_newmethod --epoch 39 --maml_nums 4 --model BAN --feat_dim 64
#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --input saved_models/rad_BAN_maml_84_pytorch_t5_other_3shot_newmethod --epoch 39 --maml_nums 5 --model BAN --feat_dim 64
#
#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --input saved_models/rad_BAN_maml_dae_84_pytorch_t1_other_3shot_newmethod --epoch 39 --maml_nums 1 --model BAN --feat_dim 64 --autoencoder
#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --input saved_models/rad_BAN_maml_dae_84_pytorch_t2_other_3shot_newmethod --epoch 39 --maml_nums 2 --model BAN --feat_dim 64 --autoencoder
#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --input saved_models/rad_BAN_maml_dae_84_pytorch_t3_other_3shot_newmethod --epoch 39 --maml_nums 3 --model BAN --feat_dim 64 --autoencoder
#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --input saved_models/rad_BAN_maml_dae_84_pytorch_t4_other_3shot_newmethod --epoch 39 --maml_nums 4 --model BAN --feat_dim 64 --autoencoder
#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --input saved_models/rad_BAN_maml_dae_84_pytorch_t5_other_3shot_newmethod --epoch 39 --maml_nums 5 --model BAN --feat_dim 64 --autoencoder

#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --output saved_models/BAN_maml_84_pytorch_t1_other_optimization_5shot_newmethod --maml_nums 1 --model BAN --feat_dim 32
#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --output saved_models/BAN_maml_84_pytorch_t2_other_optimization_5shot_newmethod --maml_nums 2 --model BAN --feat_dim 32
#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --output saved_models/BAN_maml_84_pytorch_t3_other_optimization_5shot_newmethod --maml_nums 3 --model BAN --feat_dim 32
#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --output saved_models/BAN_maml_84_pytorch_t4_other_optimization_5shot_newmethod --maml_nums 4 --model BAN --feat_dim 32
#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --output saved_models/BAN_maml_84_pytorch_t5_other_optimization_5shot_newmethod --maml_nums 5 --model BAN --feat_dim 32
#
#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --output saved_models/BAN_maml_dae_84_pytorch_t1_other_optimization_5shot_newmethod --maml_nums 1 --model BAN --feat_dim 32 --autoencoder
#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --output saved_models/BAN_maml_dae_84_pytorch_t2_other_optimization_5shot_newmethod --maml_nums 2 --model BAN --feat_dim 32 --autoencoder
#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --output saved_models/BAN_maml_dae_84_pytorch_t3_other_optimization_5shot_newmethod --maml_nums 3 --model BAN --feat_dim 32 --autoencoder
#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --output saved_models/BAN_maml_dae_84_pytorch_t4_other_optimization_5shot_newmethod --maml_nums 4 --model BAN --feat_dim 32 --autoencoder
#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --output saved_models/BAN_maml_dae_84_pytorch_t5_other_optimization_5shot_newmethod --maml_nums 5 --model BAN --feat_dim 32 --autoencoder


#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --input saved_models/BAN_maml_84_pytorch_t1_other_optimization_5shot_newmethod --epoch _best --maml_nums 1 --model BAN --feat_dim 32
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --input saved_models/BAN_maml_84_pytorch_t2_other_optimization_5shot_newmethod --epoch _best --maml_nums 2 --model BAN --feat_dim 32
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --input saved_models/BAN_maml_84_pytorch_t3_other_optimization_5shot_newmethod --epoch _best --maml_nums 3 --model BAN --feat_dim 32
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --input saved_models/BAN_maml_84_pytorch_t4_other_optimization_5shot_newmethod --epoch _best --maml_nums 4 --model BAN --feat_dim 32
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --input saved_models/BAN_maml_84_pytorch_t5_other_optimization_5shot_newmethod --epoch _best --maml_nums 5 --model BAN --feat_dim 32
#
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --input saved_models/BAN_maml_dae_84_pytorch_t1_other_optimization_5shot_newmethod --epoch _best --maml_nums 1 --model BAN --feat_dim 32 --autoencoder
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --input saved_models/BAN_maml_dae_84_pytorch_t2_other_optimization_5shot_newmethod --epoch _best --maml_nums 2 --model BAN --feat_dim 32 --autoencoder
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --input saved_models/BAN_maml_dae_84_pytorch_t3_other_optimization_5shot_newmethod --epoch _best --maml_nums 3 --model BAN --feat_dim 32 --autoencoder
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --input saved_models/BAN_maml_dae_84_pytorch_t4_other_optimization_5shot_newmethod --epoch _best --maml_nums 4 --model BAN --feat_dim 32 --autoencoder
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --input saved_models/BAN_maml_dae_84_pytorch_t5_other_optimization_5shot_newmethod --epoch _best --maml_nums 5 --model BAN --feat_dim 32 --autoencoder

#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --output saved_models/rad_BAN_maml_84_pytorch_t0-2-4_other_3shot_newmethod --maml_nums 0,2,4 --model BAN --feat_dim 64
#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --output saved_models/rad_BAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod --maml_nums 0,2,4 --model BAN --feat_dim 64 --autoencoder

#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --input saved_models/rad_BAN_maml_84_pytorch_t0-2-4_other_3shot_newmethod --epoch 39 --maml_nums 0,2,4 --model BAN --feat_dim 64
#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --input saved_models/rad_BAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod --epoch 39 --maml_nums 0,2,4 --model BAN --feat_dim 64 --autoencoder

#--------------------------------
#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --output saved_models/rad_BAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.001 --maml_nums 0,2,4 --model BAN --feat_dim 64 --autoencoder --lr 0.001
#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --input saved_models/rad_BAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.001 --epoch 39 --maml_nums 0,2,4 --model BAN --feat_dim 64 --autoencoder
#
#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --output saved_models/rad_BAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.01 --maml_nums 0,2,4 --model BAN --feat_dim 64 --autoencoder --lr 0.01
#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --input saved_models/rad_BAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.01 --epoch 39 --maml_nums 0,2,4 --model BAN --feat_dim 64 --autoencoder
#
#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --output saved_models/rad_BAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.05 --maml_nums 0,2,4 --model BAN --feat_dim 64 --autoencoder --lr 0.05
#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --input saved_models/rad_BAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.05 --epoch 39 --maml_nums 0,2,4 --model BAN --feat_dim 64 --autoencoder
#
#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --output saved_models/rad_BAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.0005 --maml_nums 0,2,4 --model BAN --feat_dim 64 --autoencoder --lr 0.0005
#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --input saved_models/rad_BAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.0005 --epoch 39 --maml_nums 0,2,4 --model BAN --feat_dim 64 --autoencoder

#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --output saved_models/rad_BAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.0001 --maml_nums 0,2,4 --model BAN --feat_dim 64 --autoencoder --lr 0.0001
#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --input saved_models/rad_BAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.0001 --epoch 39 --maml_nums 0,2,4 --model BAN --feat_dim 64 --autoencoder

#-------------------------
#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --output saved_models/rad_BAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.001_ae0.01 --maml_nums 0,2,4 --model BAN --feat_dim 64 --autoencoder --lr 0.001 --ae_alpha 0.01
#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --input saved_models/rad_BAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.001_ae0.01 --epoch 39 --maml_nums 0,2,4 --model BAN --feat_dim 64 --autoencoder

#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --output saved_models/rad_BAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.001_ae0.1 --maml_nums 0,2,4 --model BAN --feat_dim 64 --autoencoder --lr 0.001 --ae_alpha 0.1
#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --input saved_models/rad_BAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.001_ae0.1 --epoch 39 --maml_nums 0,2,4 --model BAN --feat_dim 64 --autoencoder

#-----------------------
#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --output saved_models/rad_BAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.001_seed1024 --maml_nums 0,2,4 --model BAN --feat_dim 64 --autoencoder --lr 0.001 --seed 1024
#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --input saved_models/rad_BAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.001_seed1024 --epoch 39 --maml_nums 0,2,4 --model BAN --feat_dim 64 --autoencoder
#
#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --output saved_models/rad_BAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.001_seed2104 --maml_nums 0,2,4 --model BAN --feat_dim 64 --autoencoder --lr 0.001 --seed 2104
#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --input saved_models/rad_BAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.001_seed2104 --epoch 39 --maml_nums 0,2,4 --model BAN --feat_dim 64 --autoencoder
#
#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --output saved_models/rad_BAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.001_seed1520 --maml_nums 0,2,4 --model BAN --feat_dim 64 --autoencoder --lr 0.001 --seed 1520
#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --input saved_models/rad_BAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.001_seed1520 --epoch 39 --maml_nums 0,2,4 --model BAN --feat_dim 64 --autoencoder
#
#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --output saved_models/rad_BAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.001_seed2401 --maml_nums 0,2,4 --model BAN --feat_dim 64 --autoencoder --lr 0.001 --seed 2401
#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --input saved_models/rad_BAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.001_seed2401 --epoch 39 --maml_nums 0,2,4 --model BAN --feat_dim 64 --autoencoder
#
#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --output saved_models/rad_BAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.001_seed1342 --maml_nums 0,2,4 --model BAN --feat_dim 64 --autoencoder --lr 0.001 --seed 1342
#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --input saved_models/rad_BAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.001_seed1342 --epoch 39 --maml_nums 0,2,4 --model BAN --feat_dim 64 --autoencoder

#-------------
#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --output saved_models/rad_SAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.001_seed1024 --maml_nums 0,2,4 --model SAN --feat_dim 64 --autoencoder --lr 0.001 --seed 1024
#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --input saved_models/rad_SAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.001_seed1024 --epoch 39 --maml_nums 0,2,4 --model SAN --feat_dim 64 --autoencoder
#
#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --output saved_models/rad_SAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.001_seed2104 --maml_nums 0,2,4 --model SAN --feat_dim 64 --autoencoder --lr 0.001 --seed 2104
#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --input saved_models/rad_SAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.001_seed2104 --epoch 39 --maml_nums 0,2,4 --model SAN --feat_dim 64 --autoencoder
#
#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --output saved_models/rad_SAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.001_seed1520 --maml_nums 0,2,4 --model SAN --feat_dim 64 --autoencoder --lr 0.001 --seed 1520
#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --input saved_models/rad_SAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.001_seed1520 --epoch 39 --maml_nums 0,2,4 --model SAN --feat_dim 64 --autoencoder
#
#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --output saved_models/rad_SAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.001_seed2401 --maml_nums 0,2,4 --model SAN --feat_dim 64 --autoencoder --lr 0.001 --seed 2401
#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --input saved_models/rad_SAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.001_seed2401 --epoch 39 --maml_nums 0,2,4 --model SAN --feat_dim 64 --autoencoder
#
#python3 main.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --output saved_models/rad_SAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.001_seed1342 --maml_nums 0,2,4 --model SAN --feat_dim 64 --autoencoder --lr 0.001 --seed 1342
#python3 test.py --use_RAD --RAD_dir data_RAD --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --input saved_models/rad_SAN_maml_dae_84_pytorch_t0-2-4_other_3shot_newmethod_lr0.001_seed1342 --epoch 39 --maml_nums 0,2,4 --model SAN --feat_dim 64 --autoencoder

#------------------
#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/BAN_maml_dae_84_pytorch_t0-2-5_other_5shot_newmethod_lr0.0001 --maml_nums 0,2,5 --model BAN --feat_dim 32 --autoencoder --lr 0.0001
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_dae_84_pytorch_t0-2-5_other_5shot_newmethod_lr0.0001 --maml_nums 0,2,5 --model BAN --feat_dim 32 --autoencoder --epoch _best
#
#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/BAN_maml_dae_84_pytorch_t0-2-5_other_5shot_newmethod_lr0.0005 --maml_nums 0,2,5 --model BAN --feat_dim 32 --autoencoder --lr 0.0005
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_dae_84_pytorch_t0-2-5_other_5shot_newmethod_lr0.0005 --maml_nums 0,2,5 --model BAN --feat_dim 32 --autoencoder --epoch _best
#
#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/BAN_maml_dae_84_pytorch_t0-2-5_other_5shot_newmethod_lr0.001 --maml_nums 0,2,5 --model BAN --feat_dim 32 --autoencoder --lr 0.001
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_dae_84_pytorch_t0-2-5_other_5shot_newmethod_lr0.001 --maml_nums 0,2,5 --model BAN --feat_dim 32 --autoencoder --epoch _best
#
#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/BAN_maml_dae_84_pytorch_t0-2-5_other_5shot_newmethod_lr0.005 --maml_nums 0,2,5 --model BAN --feat_dim 32 --autoencoder --lr 0.005
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_dae_84_pytorch_t0-2-5_other_5shot_newmethod_lr0.005 --maml_nums 0,2,5 --model BAN --feat_dim 32 --autoencoder --epoch _best
#
#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/BAN_maml_dae_84_pytorch_t0-2-5_other_5shot_newmethod_lr0.01 --maml_nums 0,2,5 --model BAN --feat_dim 32 --autoencoder --lr 0.01
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_dae_84_pytorch_t0-2-5_other_5shot_newmethod_lr0.01 --maml_nums 0,2,5 --model BAN --feat_dim 32 --autoencoder --epoch _best
#
#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/BAN_maml_dae_84_pytorch_t0-2-5_other_5shot_newmethod_lr0.05 --maml_nums 0,2,5 --model BAN --feat_dim 32 --autoencoder --lr 0.05
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_dae_84_pytorch_t0-2-5_other_5shot_newmethod_lr0.05 --maml_nums 0,2,5 --model BAN --feat_dim 32 --autoencoder --epoch _best

#-----------------------------
#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/BAN_maml_dae_84_pytorch_t0-2-5_other_5shot_newmethod_lr0.01_seed1024 --maml_nums 0,2,5 --model BAN --feat_dim 32 --autoencoder --lr 0.01 --seed 1024
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_dae_84_pytorch_t0-2-5_other_5shot_newmethod_lr0.01_seed1024 --maml_nums 0,2,5 --model BAN --feat_dim 32 --autoencoder --epoch _best
#
#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/BAN_maml_dae_84_pytorch_t0-2-5_other_5shot_newmethod_lr0.01_seed2104 --maml_nums 0,2,5 --model BAN --feat_dim 32 --autoencoder --lr 0.01 --seed 2104
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_dae_84_pytorch_t0-2-5_other_5shot_newmethod_lr0.01_seed2104 --maml_nums 0,2,5 --model BAN --feat_dim 32 --autoencoder --epoch _best
#
#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/BAN_maml_dae_84_pytorch_t0-2-5_other_5shot_newmethod_lr0.01_seed1520 --maml_nums 0,2,5 --model BAN --feat_dim 32 --autoencoder --lr 0.01 --seed 1520
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_dae_84_pytorch_t0-2-5_other_5shot_newmethod_lr0.01_seed1520 --maml_nums 0,2,5 --model BAN --feat_dim 32 --autoencoder --epoch _best
#
#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/BAN_maml_dae_84_pytorch_t0-2-5_other_5shot_newmethod_lr0.01_seed2401 --maml_nums 0,2,5 --model BAN --feat_dim 32 --autoencoder --lr 0.01 --seed 2401
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_dae_84_pytorch_t0-2-5_other_5shot_newmethod_lr0.01_seed2401 --maml_nums 0,2,5 --model BAN --feat_dim 32 --autoencoder --epoch _best
#
#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/BAN_maml_dae_84_pytorch_t0-2-5_other_5shot_newmethod_lr0.01_seed1342 --maml_nums 0,2,5 --model BAN --feat_dim 32 --autoencoder --lr 0.01 --seed 1342
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_dae_84_pytorch_t0-2-5_other_5shot_newmethod_lr0.01_seed1342 --maml_nums 0,2,5 --model BAN --feat_dim 32 --autoencoder --epoch _best

#-------------------------------
#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/BAN_maml_dae_84_pytorch_t1-2-5_other_5shot_newmethod_lr0.001_seed2104 --maml_nums 1,2,5 --model BAN --feat_dim 32 --autoencoder --lr 0.001 --seed 2104
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_dae_84_pytorch_t1-2-5_other_5shot_newmethod_lr0.001_seed2104 --maml_nums 1,2,5 --model BAN --feat_dim 32 --autoencoder --epoch _best

#------------------
#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/BAN_maml_84_pytorch_t-1_other_optimization_5shot_newmethod_seed1024_lr0.0001 --maml_nums -1 --model BAN --feat_dim 32 --lr 0.0001 --seed 1024 --epochs 10
#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/BAN_maml_dae_84_pytorch_t-1_other_optimization_5shot_newmethod_seed1024_lr0.0001 --maml_nums -1 --model BAN --feat_dim 32 --autoencoder --lr 0.0001 --seed 1024 --epochs 10

#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/SAN_maml_84_pytorch_t-1_other_optimization_5shot_newmethod_seed2104_lr0.001 --maml_nums -1 --model SAN --feat_dim 32 --lr 0.001 --seed 2104 --epochs 5
#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/SAN_maml_dae_84_pytorch_t-1_other_optimization_5shot_newmethod_seed1024_lr0.0001 --maml_nums -1 --model SAN --feat_dim 32 --autoencoder --lr 0.0001 --seed 1024 --epochs 5

#------------------
#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/SAN_maml_dae_84_pytorch_t0-2-5_other_optimization_5shot_newmethod_lr0.01_seed2104 --maml_nums 0,2,5 --model SAN --feat_dim 32 --autoencoder --lr 0.01 --seed 2104
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/SAN_maml_dae_84_pytorch_t0-2-5_other_optimization_5shot_newmethod_lr0.01_seed2104 --maml_nums 0,2,5 --model SAN --feat_dim 32 --autoencoder --epoch _best

#-------------
#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/BAN_maml_dae_84_pytorch_t1-3_other_5shot_newmethod_lr0.01_seed1342 --maml_nums 1,3 --model BAN --feat_dim 32 --autoencoder --lr 0.01 --seed 1342
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_dae_84_pytorch_t1-3_other_5shot_newmethod_lr0.01_seed1342 --maml_nums 1,3 --model BAN --feat_dim 32 --autoencoder --epoch 12

#--------
##python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/BAN_maml_84_pytorch_t0-2-5_other_optimization_5shot_newmethod_lr0.0001 --maml_nums 0,2,5 --model BAN --feat_dim 32 --lr 0.0001
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_84_pytorch_t0-2-5_other_optimization_5shot_newmethod_lr0.0001 --maml_nums 0,2,5 --model BAN --feat_dim 32 --epoch _best

#-----------
#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/BAN_maml_84_pytorch_t1-3_other_optimization_5shot_newmethod_lr0.0001 --maml_nums 1,3 --model BAN --feat_dim 32 --lr 0.0001
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_84_pytorch_t1-3_other_optimization_5shot_newmethod_lr0.0001 --maml_nums 1,3 --model BAN --feat_dim 32 --epoch 6

#------------
#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/BAN_maml_84_pytorch_t1_other_optimization_5shot_newmethod_lr0.0001 --maml_nums 1 --model BAN --feat_dim 32 --lr 0.0001 --epochs 5
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_84_pytorch_t1_other_optimization_5shot_newmethod_lr0.0001 --maml_nums 1 --model BAN --feat_dim 32 --epoch 2

#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/BAN_maml_84_pytorch_t2_other_optimization_5shot_newmethod_lr0.0001 --maml_nums 2 --model BAN --feat_dim 32 --lr 0.0001 --epochs 5
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_84_pytorch_t2_other_optimization_5shot_newmethod_lr0.0001 --maml_nums 2 --model BAN --feat_dim 32 --epoch 2
#
#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/BAN_maml_84_pytorch_t3_other_optimization_5shot_newmethod_lr0.0001 --maml_nums 3 --model BAN --feat_dim 32 --lr 0.0001 --epochs 5
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_84_pytorch_t3_other_optimization_5shot_newmethod_lr0.0001 --maml_nums 3 --model BAN --feat_dim 32 --epoch 2

#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/BAN_maml_84_pytorch_t4_other_optimization_5shot_newmethod_lr0.0001 --maml_nums 4 --model BAN --feat_dim 32 --lr 0.0001 --epochs 5
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_84_pytorch_t4_other_optimization_5shot_newmethod_lr0.0001 --maml_nums 4 --model BAN --feat_dim 32 --epoch 3

#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/BAN_maml_84_pytorch_t5_other_optimization_5shot_newmethod_lr0.0001 --maml_nums 5 --model BAN --feat_dim 32 --lr 0.0001 --epochs 5
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_84_pytorch_t5_other_optimization_5shot_newmethod_lr0.0001 --maml_nums 5 --model BAN --feat_dim 32 --epoch 2

#--------------
#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/BAN_maml_dae_84_pytorch_t1_other_optimization_5shot_newmethod_lr0.0001 --maml_nums 1 --model BAN --feat_dim 32 --lr 0.0001 --epochs 5 --autoencoder
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_dae_84_pytorch_t1_other_optimization_5shot_newmethod_lr0.0001 --maml_nums 1 --model BAN --feat_dim 32 --epoch 3 --autoencoder

#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/BAN_maml_dae_84_pytorch_t2_other_optimization_5shot_newmethod_lr0.0001 --maml_nums 2 --model BAN --feat_dim 32 --lr 0.0001 --epochs 5 --autoencoder
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_dae_84_pytorch_t2_other_optimization_5shot_newmethod_lr0.0001 --maml_nums 2 --model BAN --feat_dim 32 --epoch 3 --autoencoder

#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/BAN_maml_dae_84_pytorch_t4_other_optimization_5shot_newmethod_lr0.0001 --maml_nums 4 --model BAN --feat_dim 32 --lr 0.0001 --epochs 5 --autoencoder
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_dae_84_pytorch_t4_other_optimization_5shot_newmethod_lr0.0001 --maml_nums 4 --model BAN --feat_dim 32 --epoch 3 --autoencoder

#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/BAN_maml_dae_84_pytorch_t5_other_optimization_5shot_newmethod_lr0.0001 --maml_nums 5 --model BAN --feat_dim 32 --lr 0.0001 --epochs 5 --autoencoder
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_dae_84_pytorch_t5_other_optimization_5shot_newmethod_lr0.0001 --maml_nums 5 --model BAN --feat_dim 32 --epoch 3 --autoencoder

#--------
#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/BAN_maml_dae_84_pytorch_t0-2-4-5_other_optimization_5shot_newmethod_lr0.01_seed2104 --maml_nums 0,2,4,5 --model BAN --feat_dim 32 --lr 0.01 --autoencoder --seed 2104
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_dae_84_pytorch_t0-2-4-5_other_optimization_5shot_newmethod_lr0.01_seed2104 --maml_nums 0,2,4,5 --model BAN --feat_dim 32 --epoch _best --autoencoder

#---------------
#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/BAN_maml_84_pytorch_t6_other_optimization_5shot_newmethod_lr0.0001 --maml_nums 6 --model BAN --feat_dim 32 --lr 0.0001 --epochs 5
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_84_pytorch_t6_other_optimization_5shot_newmethod_lr0.0001 --maml_nums 6 --model BAN --feat_dim 32 --epoch 2

#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/BAN_maml_84_pytorch_t7_other_optimization_5shot_newmethod_lr0.0001 --maml_nums 7 --model BAN --feat_dim 32 --lr 0.0001 --epochs 5
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_84_pytorch_t7_other_optimization_5shot_newmethod_lr0.0001 --maml_nums 7 --model BAN --feat_dim 32 --epoch 4

#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/BAN_maml_dae_84_pytorch_t6_other_optimization_5shot_newmethod_lr0.0001 --maml_nums 6 --model BAN --feat_dim 32 --lr 0.0001 --epochs 5 --autoencoder
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_dae_84_pytorch_t6_other_optimization_5shot_newmethod_lr0.0001 --maml_nums 6 --model BAN --feat_dim 32 --epoch 3 --autoencoder
#
#python3 main.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --output saved_models/BAN_maml_dae_84_pytorch_t7_other_optimization_5shot_newmethod_lr0.0001 --maml_nums 7 --model BAN --feat_dim 32 --lr 0.0001 --epochs 5 --autoencoder
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_dae_84_pytorch_t7_other_optimization_5shot_newmethod_lr0.0001 --maml_nums 7 --model BAN --feat_dim 32 --epoch 3 --autoencoder


#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_84_pytorch_t0_other_optimization_5shot_seed1024_lr0.0001 --maml_nums 0 --model BAN --feat_dim 32 --epoch 1
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_84_pytorch_t1_other_optimization_5shot_newmethod --maml_nums 1 --model BAN --feat_dim 32 --epoch 0
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_84_pytorch_t1-3_other_optimization_5shot_newmethod_lr0.0001 --maml_nums 1,3 --model BAN --feat_dim 32 --epoch 6
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_84_pytorch_t0-2-5_other_optimization_5shot_newmethod_lr0.0001 --maml_nums 0,2,5 --model BAN --feat_dim 32 --epoch _best
#python3 test.py --use_RAD --RAD_dir data/converted_data --maml --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth --input saved_models/BAN_maml_84_pytorch_t0-2-3-5_other_optimization_5shot_newmethod_lr0.0001 --maml_nums 0,2,3,5 --model BAN --feat_dim 32 --epoch _best