export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 tools/train.py -c configs/rtdetr/gmdetr_r50vd_6x_align_flir_X.yml