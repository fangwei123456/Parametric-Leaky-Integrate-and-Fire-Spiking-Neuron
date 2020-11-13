# Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron

[中文README](./README_cn.md)

This  repository contains the origin codes and TensorBoard logs for the paper *Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks*. The trained models are too large that we can't upload them to this repository. But we used a identical seed during training, and we can ensure that the user can get almost the same accuracy when using our codes to train.

## Directory structure

`codes` contains the origin codes:

`models.py` defines the networks.

`train.py` trains models on the training set, tests on the test set alternately, and records the maximum test accuracy, which is the accuracy-A in the paper.

`train_val.py` splits the origin training set into a new training set and validation set, trains on the new training set, tests on the validation set alternately, and records the test accuracy on the test set only once, with the model achieving the maximum validation accuracy, which is the accuracy-B in the paper.

`logs` contains `A` and `B` directories, which contains TensorBoard logs for different accuracies, respectively.

## Setup

The origin codes uses the old version SpikingJelly. To maximize reproducibility, the user can download the latest SpikingJelly and rollback to the version that we used to train:


```bash
git clone https://github.com/fangwei123456/spikingjelly.git
cd spikingjelly
git reset --hard f13b80538042a565b0764df195594e3ee5b54255
python setup.py install
```

## Running codes

Here are the origin running for accuracy-B:

| Dataset       | Running codes                                                |
| ------------- | ------------------------------------------------------------ |
| MNIST         | python ./codes/train_val.py -init_tau 2.0 -use_max_pool -use_plif -device cuda:0 -dataset_name MNIST -log_dir_prefix /userhome/plif_test/logsd -T 8 -max_epoch 1024 -detach_reset |
| Fashion-MNIST | python ./codes/train_val.py -init_tau 2.0 -use_max_pool -use_plif -device cuda:0 -dataset_name FashionMNIST -log_dir_prefix /userhome/plif_test/logsd -T 8 -max_epoch 1024 -detach_reset |
| CIFAR10       | python ./codes/train_val.py -init_tau 2.0 -use_max_pool -use_plif -device cuda:0 -dataset_name CIFAR10 -log_dir_prefix /userhome/plif_test/logsd -T 8 -max_epoch 1024 -detach_reset |
| N-MNIST       | python ./codes/train_val.py -init_tau 2.0 -use_max_pool -device cuda:0 -dataset_name NMNIST -log_dir_prefix /userhome/plif_test/logsd -T 20 -max_epoch 1024 -detach_reset -channels 128 -number_layer 2 -split_by number -normalization None -use_plif |
| CIFAR10-DVS   | python ./codes/train_val.py -init_tau 2.0 -use_max_pool -device cuda:0 -dataset_name CIFAR10DVS -log_dir_prefix /userhome/plif_test/logsd -T 20 -max_epoch 1024 -detach_reset -channels 128 -number_layer 4 -split_by number -normalization None -use_plif |
| DVS Gesture   | python ./codes/train_val.py -init_tau 2.0 -use_max_pool -device cuda:0 -dataset_name DVS128Gesture -log_dir_prefix /userhome/plif_test/logsd -T 20 -max_epoch 1024 -detach_reset -channels 128 -number_layer 5 -split_by number -normalization None -use_plif |

