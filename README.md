# Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron

[中文README](./README_cn.md)

This  repository contains the origin codes and TensorBoard logs for the paper *[Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks](https://arxiv.org/abs/2007.05785)*. The trained models are too large that we can't upload them to this repository. But we used a identical seed during training, and we can ensure that the user can get almost the same accuracy when using our codes to train.

## Accuracy

This table shows the accuracy of using PLIF neurons, tau_0=2 and max pooling:

|            | MNIST  | Fashion-MNIST | CIFAR10 | N-MNIST | CIFAR10-DVS | DVS128 Gesture |
| ---------- | ------ | ------------- | ------- | ------- | ----------- | -------------- |
| accuracy-A | 97.72% | 94.38%        | 93.50%  | 99.61%  | 74.80%      | 97.57%         |
| accuracy-B | 99.63% | 93.85%        | 92.58%  | 99.57%  | 69.00%      | 96.53%         |

This table shows the accuracy-A of using PLIF/LIF neurons, different tau/tau_0 and average/max pooling:

|               | pooling | MNIST  | Fashion-MNIST | CIFAR-10 | N-MNIST | CIFAR10-DVS | DVS128 Gesture |
| ------------- | ------- | ------ | ------------- | -------- | ------- | ----------- | -------------- |
| PLIF,tau_0=2  | max     | 99.72% | 94.38%        | 93.5%    | 99.61%  | 74.8%       | 97.57%         |
| PLIF,tau_0=16 | max     | 99.73% | 94.65%        | 93.23%   | 99.53%  | 70.5%       | 92.01%         |
| LIF,tau=2     | max     | 99.69% | 94.17%        | 93.03%   | 99.64%  | 73.6%       | 96.88%         |
| LIF,tau=16    | max     | 99.49% | 94.47%        | 47.5%    | 99.15%  | 62.4%       | 76.74%         |
| PLIF,tau_0=2  | avg     | 99.71% | 94.74%        | 93.3%    | 99.66%  | 72.7%       | 97.22%         |

## Directory Structure

`codes` contains the origin codes:

`models.py` defines the networks.

`train.py` trains models on the training set, tests on the test set alternately, and records the maximum test accuracy, which is the accuracy-A in the paper.

`train_val.py` splits the origin training set into a new training set and validation set, trains on the new training set, tests on the validation set alternately, and records the test accuracy on the test set only once, with the model achieving the maximum validation accuracy, which is the accuracy-B in the paper.

`logs` contains `A` and `B` directories, which contains TensorBoard logs for different accuracies, respectively.

## Dependency

The origin codes uses the old version SpikingJelly. To maximize reproducibility, the user can download the latest SpikingJelly and rollback to the version that we used to train:


```bash
git clone https://github.com/fangwei123456/spikingjelly.git
cd spikingjelly
git reset --hard 73f94ab983d0167623015537f7d4460b064cfca1
python setup.py install
```

Here is the commit information:

```bash
commit 73f94ab983d0167623015537f7d4460b064cfca1
Author: fangwei123456 <fangwei123456@pku.edu.cn>
Date:   Wed Sep 30 16:42:25 2020 +0800

    增加detach reset的选项
```


## Datasets

The line 64 of `train.py`, and line 84 of `train_val.py` defines the dataset path:

`dataset_dir = '/userhome/datasets/' + dataset_name`

where `/userhome/datasets/` is the root path of all datasets.

The root path of all datasets should have the following directory structure:

```
|-- CIFAR10
|   |-- cifar-10-batches-py
|   `-- cifar-10-python.tar.gz
|-- CIFAR10DVS
|   |-- airplane.zip
|   |-- automobile.zip
|   |-- bird.zip
|   |-- cat.zip
|   |-- deer.zip
|   |-- dog.zip
|   |-- events
|   |-- frames_num_20_split_by_number_normalization_None
|   |-- frog.zip
|   |-- horse.zip
|   |-- ship.zip
|   `-- truck.zip
|-- DVS128Gesture
|   |-- DvsGesture.tar.gz
|   |-- LICENSE.txt
|   |-- README.txt
|   |-- events_npy
|   |-- extracted
|   |-- frames_num_20_split_by_number_normalization_None
|   `-- gesture_mapping.csv
|-- FashionMNIST
|   |-- FashionMNIST
|-- MNIST
|   `-- MNIST
`-- NMNIST
    |-- Test.zip
    |-- Train.zip
    |-- events
    `-- frames_num_10_split_by_number_normalization_None
```

MNIST, Fashion-MNIST and CIFAR10 dataset can be available from [torchvision](https://github.com/pytorch/vision). For neuromorphic datasets' installation, see

https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.4/spikingjelly.datasets.html

## Running codes

Here are the origin running codes for accuracy-B:

| Dataset        | Running codes                                                |
| -------------- | ------------------------------------------------------------ |
| MNIST          | python ./codes/train_val.py -init_tau 2.0 -use_max_pool -use_plif -device cuda:0 -dataset_name MNIST -log_dir_prefix /userhome/plif_test/logsd -T 8 -max_epoch 1024 -detach_reset |
| Fashion-MNIST  | python ./codes/train_val.py -init_tau 2.0 -use_max_pool -use_plif -device cuda:0 -dataset_name FashionMNIST -log_dir_prefix /userhome/plif_test/logsd -T 8 -max_epoch 1024 -detach_reset |
| CIFAR10        | python ./codes/train_val.py -init_tau 2.0 -use_max_pool -use_plif -device cuda:0 -dataset_name CIFAR10 -log_dir_prefix /userhome/plif_test/logsd -T 8 -max_epoch 1024 -detach_reset |
| N-MNIST        | python ./codes/train_val.py -init_tau 2.0 -use_max_pool -device cuda:0 -dataset_name NMNIST -log_dir_prefix /userhome/plif_test/logsd -T 10 -max_epoch 1024 -detach_reset -channels 128 -number_layer 2 -split_by number -normalization None -use_plif |
| CIFAR10-DVS    | python ./codes/train_val.py -init_tau 2.0 -use_max_pool -device cuda:0 -dataset_name CIFAR10DVS -log_dir_prefix /userhome/plif_test/logsd -T 20 -max_epoch 1024 -detach_reset -channels 128 -number_layer 4 -split_by number -normalization None -use_plif |
| DVS128 Gesture | python ./codes/train_val.py -init_tau 2.0 -use_max_pool -device cuda:0 -dataset_name DVS128Gesture -log_dir_prefix /userhome/plif_test/logsd -T 20 -max_epoch 1024 -detach_reset -channels 128 -number_layer 5 -split_by number -normalization None -use_plif |

The code can recovery training from the interruption. It will load the exist model and continue training from the last epoch.

## Arguments Definition

This table shows the definition of all arguments:

| argument        | meaning                                                      | type                                                         | default |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------- |
| init_tau        | tau of all LIF neurons, or tau_0 of PLIF neurons             | float                                                        | -       |
| batch_size      | training batch size                                          | int                                                          | 16      |
| learning_rate   | learning rate                                                | float                                                        | 1e-3    |
| T_max           | period of the learning rate schedule                         | int                                                          | 64      |
| use_plif        | use PLIF neurons                                             | action='store_true'                                          | False   |
| alpha_learnable | if given, `alpha` in the surrogate function is learnable     | action='store_true'                                          | False   |
| use_max_pool    | if given, the network will use max pooling, else use average pooling | action='store_true'                                          | False   |
| device          | use which device to train                                    | str                                                          | -       |
| dataset_name    | use which dataset                                            | str(`MNIST`,`FashionMNIST`,`CIFAR10`,`NMNIST`,`CIFAR10DVS`or`DVSGesture`) | -       |
| log_dir_prefix  | the path for TensorBoard to save logs                        | str                                                          | -       |
| T               | simulating time-step                                         | int                                                          | -       |
| channels        | the out channels of Conv2d for neuromorphic datasets         | int                                                          | -       |
| number_layer    | the number of Conv2d layers for neuromorphic datasets        | int                                                          | -       |
| split_by        | how to split the events to integrate them to frames          | str(`time` or`number` )                                      | -       |
| normalization   | normalization for frames during being integrated             | str(`frequency`,`max`,`norm`,`sum` or`None`)                 | -       |
| max_epoch       | maximum training epoch                                       | int                                                          | -       |
| detach_reset    | whether detach the voltage reset during backward             | action='store_true'                                          | False   |

For more details about `split_by`和`normalization`, see

https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.4/spikingjelly.datasets.html#integrate-events-to-frames-init-en

## New Implement

SpkingJelly (0.0.0.0.12 or the latest version) has added the network with LIF/max-pooling as an example: 

0.0.0.0.12: https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.12/clock_driven_en/14_classify_dvsg.html

latest: https://spikingjelly.readthedocs.io/zh_CN/latest/activation_based_en/classify_dvsg.html

The codes are written by the new version of SpikingJelly, which are faster than codes in this repository. 

All networks in this paper are available at SpikingJelly: 

0.0.0.0.12: https://github.com/fangwei123456/spikingjelly/blob/0.0.0.0.12/spikingjelly/clock_driven/model/parametric_lif_net.py

latest: https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/activation_based/model/sew_resnet.py

## Cite

```
@InProceedings{Fang_2021_ICCV,
    author    = {Fang, Wei and Yu, Zhaofei and Chen, Yanqi and Masquelier, Timothee and Huang, Tiejun and Tian, Yonghong},
    title     = {Incorporating Learnable Membrane Time Constant To Enhance Learning of Spiking Neural Networks},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {2661-2671}
}
```
