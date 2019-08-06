# ResNet50 example

This example demonstrates how to train a Neural Module which implements ResNet50
network on ImageNet data using multi-GPU (but single node) training.

1) *Step 1*: Get ImageNet data in [Image Folder format](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder).
2) *Step 2*: Run training. (This will also start evaluation in parallel)
```bash
python -m torch.distributed.launch --nproc_per_node=2 resnet50.py --data_root=/mnt/D1/Data/ImageNet/ImageFolder/ --num_gpus=2
```
note that `nproc_per_node` should be equal to `num_gpus`. 
If you run out of GPU memory, reduce `batch_size` parameter. This parameters is per GPU. 
3) *Step 3*: Monitor training with TensorBoard
```bash
tensorboard --logdir=resnet50
```