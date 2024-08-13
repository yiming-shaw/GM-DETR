export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 tools/train_X.py -c /data0/XYM/GM-DETR_pytorch/configs/rtdetr/gmdetr_r101vd_6x_align_flir_X.yml

torchrun --nproc_per_node=2 tools/train_X.py -c /data0/XYM/GM-DETR_pytorch/configs/gmdetr/gmdetr_r50vd_6x_align_flir_X_train1.yml

torchrun --nproc_per_node=2 tools/train_X.py -c /data0/XYM/GM-DETR_pytorch/configs/gmdetr/gmdetr_r50vd_6x_align_flir_X_train2.yml \
-r /data0/XYM/GM-DETR_pytorch/output/gmdetr_r50vd_6x_align_flir_X_train1/checkpoint0019.pth