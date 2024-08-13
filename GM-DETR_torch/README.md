## Quick start

<details>
<summary>Install</summary>

```bash
pip install -r requirements.txt
```

</details>


<details>
<summary>Data</summary>

It's same as the paddle version.

- Modify config [`img_folder`, `ann_file`](configs/dataset/coco_detection.yml)
</details>



<details>
<summary>Training & Evaluation</summary>

- Training on a Single GPU:

```shell
# training on single-gpu
export CUDA_VISIBLE_DEVICES=0
python tools/train_X.py -c configs/gmdetr/gmdetr_r101vd_6x_align_flir_X_train2.yml
```

- Training on Multiple GPUs:

```shell
# train on multi-gpu
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 tools/train_X.py -c configs/gmdetr/gmdetr_r50vd_6x_align_flir_X_train1.yml

torchrun --nproc_per_node=2 tools/train_X.py -c configs/gmdetr/gmdetr_r50vd_6x_align_flir_X_train2.yml \
-r output/gmdetr_r50vd_6x_align_flir_X_train1/xxxxx.pth
```

- Evaluation on single GPU:

```shell
# val on multi-gpu
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/gmdetr/gmdetr_r101vd_6x_align_flir_X_train2.yml -r path/to/checkpoint --test-only
```

</details>
