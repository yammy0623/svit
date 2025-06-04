# SViT: Revisiting Token Pruning for Object Detection and Instance Segmentation

This repository contains code for the paper "[Revisiting Token Pruning for Object Detection and Instance Segmentation
](https://arxiv.org/abs/2306.07050)".




## Preparation
recommended environment:
```
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html
pip install timm==0.4.12
pip install mmdet==2.28.1
pip install scipy
pip install setuptools
cd ops & sh make.sh # compile deformable attention
```

## SELF REPRODUCE NOTE
### deit_tiny_patch16_224-a1311bcf.pth is missing
Download from https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth and put it at pretrained

### Reproduce Result
#### Bounding Box
| Metrics                  | r = 0  | r = 10 | r = 20 | r = 30 | r = 50 | 
| -------------------- | ------ | ------ | ------ | ------ | ------ |
| AP (IoU=0.50:0.95)   | 0.458  | 0.458  | 0.457  | 0.456  | 0.457  |
| AP (IoU=0.50)        | 0.670  | 0.670  | 0.669  | 0.667  | 0.668  |
| AP (IoU=0.75)        | 0.501  | 0.501  | 0.499  | 0.497  | 0.499  |
| AP (small)           | 0.303  | 0.303  | 0.305  | 0.305  | 0.305  |
| AP (medium)          | 0.486  | 0.486  | 0.487  | 0.484  | 0.486  |
| AP (large)           | 0.597  | 0.597  | 0.597  | 0.598  | 0.599  |
| AR (maxDets=1000)    | 0.593  | 0.593  | 0.592  | 0.592  | 0.592  |
| AR (small)           | 0.425  | 0.425  | 0.421  | 0.421  | 0.421  |
| AR (medium)          | 0.624  | 0.624  | 0.624  | 0.624  | 0.624  |
| AR (large)           | 0.743  | 0.743  | 0.743  | 0.743  | 0.744  |
| Elapsed Time (seconds) | 407.00 | 668.22 | 647.12 | 644.67 | 643.36 | 

#### Segmentation
| Metrics                   | r = 0  | r = 10 | r = 20 | r = 30 | r = 50 | 
| -------------------- | ------ | ------ | ------ | ------ | ------ |
| AP (IoU=0.50:0.95)   | 0.458  | 0.409  | 0.409  | 0.409  | 0.410  |
| AP (IoU=0.50)        | 0.670  | 0.639  | 0.640  | 0.640  | 0.641  |
| AP (IoU=0.75)        | 0.439  | 0.439  | 0.438  | 0.438  | 0.439  |
| AP (small)           | 0.303  | 0.220  | 0.220  | 0.221  | 0.221  |
| AP (medium)          | 0.486  | 0.436  | 0.436  | 0.436  | 0.436  |
| AP (large)           | 0.597  | 0.595  | 0.594  | 0.594  | 0.595  |
| AR (maxDets=1000)    | 0.593  | 0.535  | 0.535  | 0.535  | 0.536  |
| AR (small)           | 0.425  | 0.360  | 0.360  | 0.360  | 0.361  |
| AR (medium)          | 0.624  | 0.570  | 0.570  | 0.570  | 0.570  |
| AR (large)           | 0.743  | 0.701  | 0.701  | 0.701  | 0.701  |
| Elapsed Time (seconds) | 407.00 | 680.85 | 659.73 | 656.67 | 655.40 | 

## Docker Run
```
docker run -it --name svit_container_new --runtime=nvidia --gpus all --shm-size=16g \
    --device=/dev/nvidia-uvm \
    --device=/dev/nvidia-uvm-tools \
    --device=/dev/nvidia-modeset \
    --device=/dev/nvidiactl \
    --device=/dev/nvidia0 \
    --device=/dev/nvidia1 \
    --device=/dev/nvidia2 \
    --device=/dev/nvidia3 \
    -v /tmp2/christine/svit:/workspace \
    -v /home/christine:/home \
    complete_image /bin/bash
```

## Data Preparation

Please prepare COCO according to the guidelines in [MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md). <br/>
Alternatively, download [download_dataset.py](https://github.com/open-mmlab/mmdetection/blob/master/tools/misc/download_dataset.py) and run ```python download_dataset.py --dataset-name coco2017```.<br/>
The dataset should have (or symlinked to have) the following folder structure:
```text
root_folder
├── mmcv_custom
├── mmdet_custom
├── configs
├── ops
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
```


## Results

The following models exploit [Mask R-CNN](./configs/mask_rcnn/) and use [ViT-Adapter](https://github.com/czczup/ViT-Adapter) as backbones, which adapt [DeiT](https://github.com/facebookresearch/deit) without windows in this repo, since token pruning is incompatible with windows. Token pruning is introduced in the fintuing after the dense model has been trained for 36 epochs, and the finetuning contains 4 or 6 epochs.

| Backbone        | Pre-train                                                                             | Lr schd | box AP | mask AP | #Param |  FPS  | Config                                                        | Download                                                                                     |   Logs           |
|:---------------:|:-------------------------------------------------------------------------------------:|:-------:|:------:|:-------:|:------:|:-----:|:-------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|:----------------:|
| ViT-Adapter-T   | [DeiT-T](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth)      | 3x+MS   | 45.8   | 40.9    | 28M    | 18.45 | [config](./configs/mask_rcnn/vit-adapter-t-3x.py)             | [model](https://drive.google.com/file/d/1CjCTDauTXUc-FauAvqtc7aitFo4GfK_z/view?usp=drive_link)  |      [log](https://drive.google.com/file/d/1r_R-xmaCjst6yQPFVROfU8Xlu3-HHW_R/view?usp=drive_link)     |
| SViT-Adapter-T  | [ViT-Adapter-T](https://drive.google.com/file/d/1CjCTDauTXUc-FauAvqtc7aitFo4GfK_z/view?usp=drive_link)                                                                     | 0.5x+MS   | 45.5   | 40.7    | 28M    | 22.32 |[config](./configs/mask_rcnn/svit-adapter-t-0.5x-ftune.py)       | [model](https://drive.google.com/file/d/1jpslNG16YBLe6h7E1_ZYmCSxYG86Euup/view?usp=drive_link)  |      [log](https://drive.google.com/file/d/11LdhT7Js6PCIlAhI1AHAGY6pL0xVGxLF/view?usp=drive_link)     |
| ViT-Adapter-S   | [DeiT-S](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth)     | 3x+MS   | 48.5   | 42.8    | 48M    | 11.70  | [config](./configs/mask_rcnn/vit-adapter-s-3x.py)             | [model](https://drive.google.com/file/d/1f9brYvhuoH6xFXK_nhsd7a5y9ewDIwHm/view?usp=drive_link)  |      [log](https://drive.google.com/file/d/1C5xipYNqeR84JMB0dpaTOxaaIpJ9PbTh/view?usp=drive_link)     |
| SViT-Adapter-S  | [ViT-Adapter-S](https://drive.google.com/file/d/1f9brYvhuoH6xFXK_nhsd7a5y9ewDIwHm/view?usp=drive_link)                                                                     | 0.33x+MS   | 48.2   | 42.5    | 48M    | 15.75  |[config](./configs/mask_rcnn/svit-adapter-s-0.33x-ftune.py)       | [model](https://drive.google.com/file/d/1F0WlAWQwZ8obgkUsTT2hdkTW8raVtlq_/view?usp=drive_link)  |      [log](https://drive.google.com/file/d/1gWujtDm5xGGdPFsmY9nJFzh_J-FNrrgo/view?usp=drive_link)     |

## Evaluation

To evaluate SViT-Adapter-S on COCO val2017 on a single node with 8 gpus run:

```shell
sh dist_test.sh configs/mask_rcnn/svit-adapter-s-0.33x-ftune.py pretrained/svit-adapter-s-0.33x.pth 8 --eval bbox segm
```

## Training

#### Dense Training:
To train a dense ViT-Adapter-T with global attention (Mask R-CNN) on COCO train2017 on a single node with 4 gpus for 36 epochs run:
```shell
sh dist_train.sh configs/mask_rcnn/vit-adapter-t-3x.py 4
```

To train a dense ViT-Adapter-S with global attention (Mask R-CNN) on COCO train2017 on a single node with 8 gpus for 36 epochs run:
```shell
sh dist_train.sh configs/mask_rcnn/vit-adapter-s-3x.py 8
```

The number of gpus x `samples_per_gpu` from the config file should be equal to 16.

#### Sparse Finetuning: 
To finetune the sparse SViT-Adapter-T with pruned tokens (Mask R-CNN) on COCO train2017 on a single node with 4 gpus for 6 epochs run:
```shell
sh dist_train.sh configs/mask_rcnn/svit-adapter-t-0.5x-ftune.py 4
```

To finetune the sparse SViT-Adapter-S with pruned tokens (Mask R-CNN) on COCO train2017 on a single node with 8 gpus for 4 epochs run:
```shell
sh dist_train.sh configs/mask_rcnn/svit-adapter-s-0.33x-ftune.py 8
```

The number of gpus x `samples_per_gpu` from the config file should be equal to 16.

## Speed Measurement

We provide the script to compare the models' speeds: 

```shell
python speed_test.py
```

## Image Demo
We provide the script to visualize the token pruning process:
```shell
python selector_demo.py data/coco/val2017/000000046252.jpg configs/mask_rcnn/demo-svit-adapter-s-0.33x-ftune.py pretrained/svit-adapter-s-0.33x.pth 0

python selector_demo.py data/coco/val2017/000000046252.jpg configs/mask_rcnn/demo-svit-adapter-t-0.5x-ftune.py pretrained/svit-adapter-t-0.5x.pth 0

python selector_demo.py data/coco/val2017/000000046252.jpg configs/diff_r_test_demo/tome-vit-adapter-t-3x_1000.py pretrained/vit-adapter-t-3x.pth 1000
```


<p align="center">
  <img src="https://user-images.githubusercontent.com/71677542/231611200-943499d6-43a4-4ef4-9d25-bff0f7df4599.png" width="750">
</p>


## Citation
If this work is helpful for your research, please consider citing the following BibTex entry:
```
@InProceedings{Liu_2024_WACV,
  title={Revisiting Token Pruning for Object Detection and Instance Segmentation},
  author={Liu, Yifei and Gehrig, Mathias and Messikommer, Nico and Cannici, Marco and Scaramuzza, Davide},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2024}
}
```

## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Code Acknowledgments

This project has used code from the following projects:<li>
[timm](https://github.com/huggingface/pytorch-image-models), [DeiT](https://github.com/facebookresearch/deit/blob/main/README_deit.md), [EViT](https://github.com/youweiliang/evit), [MMDetection](https://github.com/open-mmlab/mmdetection) and [ViT-Adapter](https://github.com/czczup/ViT-Adapter).</li>


# Ablation Study
## visualization and token usage
```
./demo_test.sh
```
The image will be saved to demo folder

## test performance
```
./dist_test.sh configs/mask_rcnn/tome-vit-adapter-t-3x.py pretrained/vit-adapter-t-3x.pth 1 29800 --eval bbox segm
```
The result will be saved to result folder
## test speed
```
python speed_test.py
```
The result will be appended to speed.txt
