#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c /data0/xiaoyiming/Paddledet_GM-DETR/configs/gmdetr/gmdetr_hgnetv2_x_5x_align_flir_v1_ir_X_class3_train1.yml \
--eval \
--use_vdl=true \
--vdl_log_dir=./vdl_dir/gmdetr_hgnetv2_x_5x_align_flir_v1_ir_X_class3_train1


CUDA_VISIBLE_DEVICES=0 python tools/train.py -c /data0/xiaoyiming/Paddledet_GM-DETR/configs/gmdetr/gmdetr_hgnetv2_x_5x_LLVIP_ir_X_class1_train1.yml \
--eval \
--use_vdl=true \
--vdl_log_dir=./vdl_dir/gmdetr_hgnetv2_x_5x_LLVIP_ir_X_class1_train1

CUDA_VISIBLE_DEVICES=0 python tools/train.py -c /data0/xiaoyiming/Paddledet_GM-DETR/configs/rtdetr/rtdetr_hgnetv2_x_5x_align_flir_v1_ir_class3.yml \
--eval \
--use_vdl=true \
--vdl_log_dir=./vdl_dir/rtdetr_hgnetv2_x_5x_LLVIP_ir_class1