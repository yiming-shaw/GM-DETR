#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/gmdetr/gmdetr_hgnetv2_x_5x_align_flir_v1_ir_X_class3_train2.yml --eval \
-r /output/gmdetr_hgnetv2_x_5x_align_flir_v1_ir_X_class3_train1/9.pdparams