# Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron

[English README](./README.md)

本仓库包含 *[Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks](https://arxiv.org/abs/2007.05785)* 一文的原始代码、TensorBoard日志。原始模型的体积太大，因此我们没有上传到此仓库。但我们在训练时固定了随机种子，能够确保使用者在重新训练时，得到几乎一致的性能。

## 性能

下表为使用PLIF神经元，tau_0=2，最大池化的正确率：

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

## 文件结构

`codes`文件夹包含原始代码，其中：

`models.py`定义网络

`train.py`在训练集上训练，在测试集上测试，报告的性能是最高测试集性能，对应原文中的accuracy-A

`train_val.py`将训练集重新划分成训练集和验证集，在训练集上训练，在验证集上测试，报告的性能是在验证集正确率最高时的测试集性能，对应原文中的accuracy-B

`logs`文件夹下的A目录和B目录，包含相应的TensorBoard日志

## 依赖

基于老版本的[SpikingJelly](https://github.com/fangwei123456/spikingjelly)。为确保可复现性，可以下载最新版的SpikingJelly后，再回退到原文训练时使用的版本：


```bash
git clone https://github.com/fangwei123456/spikingjelly.git
cd spikingjelly
git reset --hard 73f94ab983d0167623015537f7d4460b064cfca1
python setup.py install
```

对应的当时版本提交信息为：

```bash
commit 73f94ab983d0167623015537f7d4460b064cfca1
Author: fangwei123456 <fangwei123456@pku.edu.cn>
Date:   Wed Sep 30 16:42:25 2020 +0800

    增加detach reset的选项
```

## 数据集

`train.py`的64行，和`train_val.py`的84行定义了数据集的路径：

`dataset_dir = '/userhome/datasets/' + dataset_name`

其中`/userhome/datasets/` 为所有数据集的根目录。

根目录下应该有如下文件夹：

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

MNIST, Fashion-MNIST和CIFAR10数据集可以由 [torchvision](https://github.com/pytorch/vision)直接提供下载；神经形态数据集的安装，请参见

https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.4/spikingjelly.datasets.html

## 运行示例

这里是获得accuracy-B的运行命令:

| 数据集         | 运行命令                                                     |
| -------------- | ------------------------------------------------------------ |
| MNIST          | python ./codes/train_val.py -init_tau 2.0 -use_max_pool -use_plif -device cuda:0 -dataset_name MNIST -log_dir_prefix /userhome/plif_test/logsd -T 8 -max_epoch 1024 -detach_reset |
| Fashion-MNIST  | python ./codes/train_val.py -init_tau 2.0 -use_max_pool -use_plif -device cuda:0 -dataset_name FashionMNIST -log_dir_prefix /userhome/plif_test/logsd -T 8 -max_epoch 1024 -detach_reset |
| CIFAR10        | python ./codes/train_val.py -init_tau 2.0 -use_max_pool -use_plif -device cuda:0 -dataset_name CIFAR10 -log_dir_prefix /userhome/plif_test/logsd -T 8 -max_epoch 1024 -detach_reset |
| N-MNIST        | python ./codes/train_val.py -init_tau 2.0 -use_max_pool -device cuda:0 -dataset_name NMNIST -log_dir_prefix /userhome/plif_test/logsd -T 10 -max_epoch 1024 -detach_reset -channels 128 -number_layer 2 -split_by number -normalization None -use_plif |
| CIFAR10-DVS    | python ./codes/train_val.py -init_tau 2.0 -use_max_pool -device cuda:0 -dataset_name CIFAR10DVS -log_dir_prefix /userhome/plif_test/logsd -T 20 -max_epoch 1024 -detach_reset -channels 128 -number_layer 4 -split_by number -normalization None -use_plif |
| DVS128 Gesture | python ./codes/train_val.py -init_tau 2.0 -use_max_pool -device cuda:0 -dataset_name DVS128Gesture -log_dir_prefix /userhome/plif_test/logsd -T 20 -max_epoch 1024 -detach_reset -channels 128 -number_layer 5 -split_by number -normalization None -use_plif |

代码具有断点续训功能，如果检测到已经有保存的模型文件，会继续上一次的训练。

## 参数定义

下表解释了训练代码中各个参数的含义：

| 参数名          | 含义                                                         | 类型                                                         | 默认值 |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------ |
| init_tau        | LIF神经元的tau，或PLIF神经元的tau_0                          | float                                                        | -      |
| batch_size      | 训练时的batch大小                                            | int                                                          | 16     |
| learning_rate   | 学习率                                                       | float                                                        | 1e-3   |
| T_max           | 学习率调节器的周期                                           | int                                                          | 64     |
| use_plif        | 启用此选项，则使用PLIF神经元；否则使用LIF神经元              | action='store_true'                                          | False  |
| alpha_learnable | 启用此选项，则替代函数中的参数`alpha`是可学习的；否则是不可学习的 | action='store_true'                                          | False  |
| use_max_pool    | 启用此选项，则使用最大池化；否则使用平均池化                 | action='store_true'                                          | False  |
| device          | 使用哪个设备进行训练                                         | str                                                          | -      |
| dataset_name    | 使用哪个数据集                                               | str(`MNIST`,`FashionMNIST`,`CIFAR10`,`NMNIST`,`CIFAR10DVS`或`DVSGesture`) | -      |
| log_dir_prefix  | 保存TensorBoard日志目录的位置                                | str                                                          | -      |
| T               | 网络仿真时长                                                 | int                                                          | -      |
| channels        | 神经形态数据集的网络中卷积层所使用的输出通道数               | int                                                          | -      |
| number_layer    | 神经形态数据集的网络中卷积层的数量                           | int                                                          | -      |
| split_by        | 以哪种方式对神经形态数据进行划分，然后积分得到帧数据         | str(`time` 或`number` )                                      | -      |
| normalization   | 积分帧数据的正则化方式                                       | str(`frequency`,`max`,`norm`,`sum`或`None`)                  | -      |
| max_epoch       | 最大训练轮数                                                 | int                                                          | -      |
| detach_reset    | 是否在反向传播计算图中detach掉电压重置                       | action='store_true'                                          | False  |

关于参数`split_by`和`normalization`的更多信息，参见

https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.4/spikingjelly.datasets.html#spikingjelly.datasets.utils.integrate_events_to_frames

## 新版实现

SpikingJelly (0.0.0.0.12或 latest版本) 将本文中用于分类DVS手势的网络进行了实现，参见：

0.0.0.0.12: https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.12/clock_driven/14_classify_dvsg.html

latest: https://spikingjelly.readthedocs.io/zh_CN/latest/activation_based/classify_dvsg.html


教程中的代码使用新版的SpikingJelly实现，速度比本仓库的代码更快。

SpikingJelly实现了本文中的全部网络结构：

0.0.0.0.12: https://github.com/fangwei123456/spikingjelly/blob/0.0.0.0.12/spikingjelly/clock_driven/model/parametric_lif_net.py

latest: https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/activation_based/model/sew_resnet.py


## 引用

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
