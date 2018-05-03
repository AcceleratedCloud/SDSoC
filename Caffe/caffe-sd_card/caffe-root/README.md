# Caffe Tools


In this folder we provide the Caffe application binaries for image classification and benchmarking.
Change names accordingly to switch between CPU `ARM Cortex-A9` and FPGA `ZedBoard` implementations:

- Classification (`classification-cpu.bin` and `classification-fpga.bin`) 

- Benchmark (`caffe-cpu.bin` and `caffe-fpga.bin`)

We also include some pretrained models on `/models` folder for testing (the performance differs from model to model, image batching etc.). Last we provide CIFAR10 data in `/data/cifar10` folder to test the accuracy on CIFAR10 validation set. For runnning the following applications make sure you have already set the library path and caffe root path correctly as below.
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/mylibs/
export CAFFE_ROOT=/mnt/caffe-root/
```

### Classification

An example classification of 3 sample images (batched) with SqueezeNet model using FPGA:

1. Go to `/models/SqueezeNet_v1.1` directory
```
cd $CAFFE_ROOT/models/SqueezeNet_v1.1
```
2. Run 
```
$CAFFE_ROOT/classification-fpga.bin deploy.prototxt squeezenet_v1.1.caffemodel imagenet_mean.binaryproto synset_words.txt $CAFFE_ROOT/data/images/*.jpg
```

### Benchmarking

An example benchmark for the layer-by-layer execution on GoogleNet model using FPGA:
1. Go to `/models/bvlc_googlenet` directory
```
cd $CAFFE_ROOT/models/bvlc_googlenet
```
2. Run 
```
$CAFFE_ROOT/caffe-fpga.bin time -model deploy.prototxt -weights bvlc_googlenet.caffemodel -iterations 5
```

### Accuracy

In order to test the accuracy on CIFAR10 data using FPGA:

1. Go to `/data/cifar10` directory
```
cd $CAFFE_ROOT/data/cifar10
```

2. Run 
```
$CAFFE_ROOT/caffe-fpga.bin test -model cifar10_quick_train_test.prototxt -weights cifar10_quick_iter_5000.caffemodel
```
