NeMo CV Collection: Neural Modules for Computer Vision
====================================================================

The NeMo CV collection offers modules useful for the following computer vision applications.

For now the collection focuses only on Image Classification.

1. MNIST classification:
 * a thin DL wrapper around torchvision MNIST dataset
 * classification with the classic LeNet-5
 * classification with a graph: ReshapeTensor -> FeedForwardNetwork -> LogProbs
 * classification with a graph: ConvNet -> ReshapeTensor -> FeedForwardNetwork -> LogProbs

2. CIFAR10 classification:
 * a thin DL wrapper around torchvision CIFAR10 dataset
 * classification with a graph: ConvNet -> ReshapeTensor -> FeedForwardNetwork -> LogProbs
 * classification with a graph: ImageEncoder (ResNet-50 feature maps) -> FeedForwardNetwork -> LogProbs

3. CIFAR100 classification:
 * a thin DL wrapper around torchvision CIFAR100 dataset
 * classification with a graph: ImageEncoder (VGG-16 with FC6 reshaped) -> LogProbs

4. STL10 classification:
  * a thin DL wrapper around torchvision STL10 dataset
