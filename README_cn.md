# Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron

[English README](./README.md)

本仓库包含 *Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks* 一文的原始代码、TensorBoard日志。原始模型的体积太大，因此我们没有上传到此仓库。但我们在训练时固定了随机种子，能够确保使用者在重新训练时，得到几乎一致的性能。

## 文件结构

`codes`文件夹包含原始代码，其中：

`models.py`定义网络

`train.py`在训练集上训练，在测试集上测试，报告的性能是最高测试集性能，对应原文中的accuracy-A

`train_val.py`将训练集重新划分成训练集和验证集，在训练集上训练，在验证集上测试，报告的性能是在验证集正确率最高时的测试集性能，对应原文中的accuracy-B

`logs`文件夹下的A目录和B目录，包含相应的TensorBoard日志

## 依赖

基于老版本的SpikingJelly。为确保可复现性，可以下载最新版的SpikingJelly后，再回退到原文训练时使用的版本：


```bash
git clone https://github.com/fangwei123456/spikingjelly.git
cd spikingjelly
git reset --hard f13b80538042a565b0764df195594e3ee5b54255
python setup.py install
```

## 运行示例

训练原文中的accuracy-B对应的模型：

| 数据集        | 启动命令                                                     |
| ------------- | ------------------------------------------------------------ |
| MNIST         | python ./codes/train_val.py -init_tau 2.0 -use_max_pool -use_plif -device cuda:0 -dataset_name MNIST -log_dir_prefix /userhome/plif_test/logsd -T 8 -max_epoch 1024 -detach_reset |
| Fashion-MNIST | python ./codes/train_val.py -init_tau 2.0 -use_max_pool -use_plif -device cuda:0 -dataset_name FashionMNIST -log_dir_prefix /userhome/plif_test/logsd -T 8 -max_epoch 1024 -detach_reset |
| CIFAR10       | python ./codes/train_val.py -init_tau 2.0 -use_max_pool -use_plif -device cuda:0 -dataset_name CIFAR10 -log_dir_prefix /userhome/plif_test/logsd -T 8 -max_epoch 1024 -detach_reset |
| N-MNIST       | python ./codes/train_val.py -init_tau 2.0 -use_max_pool -device cuda:0 -dataset_name NMNIST -log_dir_prefix /userhome/plif_test/logsd -T 20 -max_epoch 1024 -detach_reset -channels 128 -number_layer 2 -split_by number -normalization None -use_plif |
| CIFAR10-DVS   | python ./codes/train_val.py -init_tau 2.0 -use_max_pool -device cuda:0 -dataset_name CIFAR10DVS -log_dir_prefix /userhome/plif_test/logsd -T 20 -max_epoch 1024 -detach_reset -channels 128 -number_layer 4 -split_by number -normalization None -use_plif |
| DVS Gesture   | python ./codes/train_val.py -init_tau 2.0 -use_max_pool -device cuda:0 -dataset_name DVS128Gesture -log_dir_prefix /userhome/plif_test/logsd -T 20 -max_epoch 1024 -detach_reset -channels 128 -number_layer 5 -split_by number -normalization None -use_plif |

