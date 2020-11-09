import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import functional
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import readline
import os
import argparse
import time
import sys

torch.backends.cudnn.benchmark = True
_seed_ = 2020
torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)
import models

'''
静态数据集
/userhome/anaconda3/envs/pytorch-env/bin/python /userhome/plif_test/train.py -init_tau 2.0 -use_plif -use_max_pool -device cuda:0 -dataset_name CIFAR10 -log_dir_prefix ./logsd -T 8 -max_epoch 1024 -detach_reset

/userhome/anaconda3/envs/pytorch-env/bin/python /userhome/plif_test/train.py -init_tau 2.0 -use_plif -use_max_pool -device cuda:0 -dataset_name CIFAR10DVS -log_dir_prefix /userhome/plif_test/logsd -T 20 -max_epoch 512 -detach_reset -channels 128 -number_layer 4 -split_by number -normalization None

'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # init_tau, batch_size, learning_rate, T_max, log_dir, use_plif
    parser.add_argument('-init_tau', type=float)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-learning_rate', type=float, default=1e-3)
    parser.add_argument('-T_max', type=int, default=64)
    parser.add_argument('-use_plif', action='store_true', default=False)
    parser.add_argument('-alpha_learnable', action='store_true', default=False)
    parser.add_argument('-use_max_pool', action='store_true', default=False)
    parser.add_argument('-device', type=str)
    parser.add_argument('-dataset_name', type=str)
    parser.add_argument('-log_dir_prefix', type=str)
    parser.add_argument('-T', type=int)
    parser.add_argument('-channels', type=int)
    parser.add_argument('-number_layer', type=int)
    parser.add_argument('-split_by', type=str)
    parser.add_argument('-normalization', type=str)
    parser.add_argument('-max_epoch', type=int)
    parser.add_argument('-detach_reset', action='store_true', default=False)

    args = parser.parse_args()
    argv = ' '.join(sys.argv)

    print(args)
    init_tau = args.init_tau
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    T_max = args.T_max
    use_plif = args.use_plif
    alpha_learnable = args.alpha_learnable
    use_max_pool = args.use_max_pool
    device = args.device
    dataset_name = args.dataset_name
    dataset_dir = '/userhome/datasets/' + dataset_name
    # dataset_dir = '/raid/fangw01/cifar10'
    log_dir_prefix = args.log_dir_prefix
    T = args.T
    max_epoch = args.max_epoch
    detach_reset = args.detach_reset

    number_layer = args.number_layer
    channels = args.channels
    split_by = args.split_by
    normalization = args.normalization
    if normalization == 'None':
        normalization = None

    if dataset_name != 'MNIST' and dataset_name != 'FashionMNIST' and dataset_name != 'CIFAR10':
        dir_name = f'{dataset_name}_init_tau_{init_tau}_use_plif_{use_plif}_use_max_pool_{use_max_pool}_T_{T}_c_{channels}_n_{number_layer}_split_by_{split_by}_normalization_{normalization}_detach_reset_{detach_reset}'
    else:
        dir_name = f'{dataset_name}_init_tau_{init_tau}_use_plif_{use_plif}_use_max_pool_{use_max_pool}_T_{T}_detach_reset_{detach_reset}'

    log_dir = os.path.join(log_dir_prefix, dir_name)

    pt_dir = os.path.join(log_dir_prefix, 'pt_' + dir_name)
    print(log_dir, pt_dir)
    if not os.path.exists(pt_dir):
        os.mkdir(pt_dir)
    class_num = 10  # 绝大多数数据集为10，如果不一样，在新建dataloader的时候再修改
    if dataset_name == 'MNIST':
        transform_train, transform_test = models.get_transforms(dataset_name)
        train_data_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.MNIST(
                root=dataset_dir,
                train=True,
                transform=transform_train,
                download=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            pin_memory=True)
        test_data_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.MNIST(
                root=dataset_dir,
                train=False,
                transform=transform_test,
                download=True),
            batch_size=batch_size * 8,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            pin_memory=True)
    elif dataset_name == 'FashionMNIST':
        transform_train, transform_test = models.get_transforms(dataset_name)
        train_data_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.FashionMNIST(
                root=dataset_dir,
                train=True,
                transform=transform_train,
                download=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            pin_memory=True)
        test_data_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.FashionMNIST(
                root=dataset_dir,
                train=False,
                transform=transform_test,
                download=True),
            batch_size=batch_size * 16,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            pin_memory=True)
    elif dataset_name == 'CIFAR10':
        transform_train, transform_test = models.get_transforms(dataset_name)

        train_data_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.CIFAR10(
                root=dataset_dir,
                train=True,
                transform=transform_train,
                download=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            pin_memory=True)
        test_data_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.CIFAR10(
                root=dataset_dir,
                train=False,
                transform=transform_test,
                download=True),
            batch_size=batch_size * 16,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            pin_memory=True)
    elif dataset_name == 'NMNIST':
        from spikingjelly.datasets.n_mnist import NMNIST

        train_data_loader = torch.utils.data.DataLoader(
            dataset=NMNIST(dataset_dir, train=True, use_frame=True, frames_num=T, split_by=split_by,
                           normalization=normalization),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            pin_memory=True)
        test_data_loader = torch.utils.data.DataLoader(
            dataset=NMNIST(dataset_dir, train=False, use_frame=True, frames_num=T, split_by=split_by,
                           normalization=normalization),
            batch_size=batch_size * 4,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            pin_memory=True)
    elif dataset_name == 'CIFAR10DVS':
        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS

        train_data_loader = torch.utils.data.DataLoader(
            dataset=CIFAR10DVS(dataset_dir, train=True, split_ratio=0.9, use_frame=True, frames_num=T,
                               split_by=split_by, normalization=normalization),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            pin_memory=True)
        test_data_loader = torch.utils.data.DataLoader(
            dataset=CIFAR10DVS(dataset_dir, train=False, split_ratio=0.9, use_frame=True, frames_num=T,
                               split_by=split_by, normalization=normalization),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            pin_memory=True)
    elif dataset_name == 'ASLDVS':
        class_num = 24
        from spikingjelly.datasets.asl_dvs import ASLDVS

        train_data_loader = torch.utils.data.DataLoader(
            dataset=ASLDVS(dataset_dir, train=True, split_ratio=0.8, use_frame=True, frames_num=T,
                           split_by=split_by, normalization=normalization),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            pin_memory=True)
        test_data_loader = torch.utils.data.DataLoader(
            dataset=ASLDVS(dataset_dir, train=False, split_ratio=0.8, use_frame=True, frames_num=T,
                           split_by=split_by, normalization=normalization),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            pin_memory=True)
    elif dataset_name == 'DVS128Gesture':
        class_num = 11
        from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

        train_data_loader = torch.utils.data.DataLoader(
            dataset=DVS128Gesture(dataset_dir, train=True, use_frame=True, frames_num=T,
                                  split_by=split_by, normalization=normalization),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            pin_memory=True)
        test_data_loader = torch.utils.data.DataLoader(
            dataset=DVS128Gesture(dataset_dir, train=False, use_frame=True, frames_num=T,
                                  split_by=split_by, normalization=normalization),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            pin_memory=True)

    check_point_path = os.path.join(pt_dir, 'check_point.pt')
    check_point_max_path = os.path.join(pt_dir, 'check_point_max.pt')

    net_max_path = os.path.join(pt_dir, 'net_max.pt')
    optimizer_max_path = os.path.join(pt_dir, 'optimizer_max.pt')
    scheduler_max_path = os.path.join(pt_dir, 'scheduler_max.pt')
    check_point = None
    if os.path.exists(check_point_path):
        check_point = torch.load(check_point_path, map_location=device)
        net = check_point['net']
        print(net.train_times, net.max_test_accuracy)
    else:
        if dataset_name == 'MNIST':
            net = models.MNISTNet(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool,
                                  alpha_learnable=alpha_learnable, detach_reset=detach_reset).to(device)
        elif dataset_name == 'FashionMNIST':
            net = models.FashionMNISTNet(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool,
                                         alpha_learnable=alpha_learnable, detach_reset=detach_reset).to(device)
        elif dataset_name == 'CIFAR10':
            net = models.Cifar10Net(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool,
                                    alpha_learnable=alpha_learnable, detach_reset=detach_reset).to(device)
        elif dataset_name == 'NMNIST':
            net = models.NMNISTNet(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool,
                                   alpha_learnable=alpha_learnable, detach_reset=detach_reset, channels=channels,
                                   number_layer=number_layer).to(device)
        elif dataset_name == 'CIFAR10DVS':
            net = models.CIFAR10DVSNet(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool,
                                       alpha_learnable=alpha_learnable, detach_reset=detach_reset, channels=channels,
                                       number_layer=number_layer).to(device)
        elif dataset_name == 'ASLDVS':
            net = models.ASLDVSNet(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool,
                                   alpha_learnable=alpha_learnable, detach_reset=detach_reset, channels=channels,
                                   number_layer=number_layer).to(device)
        elif dataset_name == 'DVS128Gesture':
            net = models.DVS128GestureNet(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool,
                                          alpha_learnable=alpha_learnable, detach_reset=detach_reset, channels=channels,
                                          number_layer=number_layer).to(device)

    print(net)
    # 使用Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    if check_point is not None:
        optimizer.load_state_dict(check_point['optimizer'])
        scheduler.load_state_dict(check_point['scheduler'])
        log_data_list = check_point['log_data_list']
        del check_point
    else:
        log_data_list = []

    if log_data_list.__len__() > 0 and not os.path.exists(log_dir):
        # 当tensorboard出现了飞线，可以人为删除log_dir，在这种情况下
        # log_data_list非空，而log_dir不存在
        # 可以从log_data_list将tensorboard重写，避免飞线
        rewrite_tb = True
    else:
        rewrite_tb = False
    writer = SummaryWriter(log_dir)
    if rewrite_tb:
        for i in range(log_data_list.__len__()):
            for item in log_data_list[i]:
                writer.add_scalar(item[0], item[1], item[2])

    if net.epoch != 0:
        net.epoch += 1
    ckpt_time = time.time()
    for net.epoch in range(net.epoch, max_epoch):
        start_time = time.time()

        log_data_list.append([])
        print(
            f'log_dir={log_dir}, max_test_accuracy={net.max_test_accuracy}, train_times={net.train_times}, epoch={net.epoch}')
        print(args)
        print(argv)

        net.train()
        for img, label in train_data_loader:
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            out_spikes_counter = net(img)
            out_spikes_counter_frequency = out_spikes_counter / net.T
            loss = F.mse_loss(out_spikes_counter_frequency, F.one_hot(label, class_num).float())
            loss.backward()
            optimizer.step()
            functional.reset_net(net)
            if net.train_times % 256 == 0:
                accuracy = (out_spikes_counter_frequency.argmax(dim=1) == label).float().mean().item()
                log_data_list[-1].append(('train_accuracy', accuracy, net.train_times))
                log_data_list[-1].append(('train_loss', loss.item(), net.train_times))
            net.train_times += 1
        scheduler.step()

        net.eval()
        with torch.no_grad():
            test_sum = 0
            correct_sum = 0
            for img, label in test_data_loader:
                img = img.to(device)
                label = label.to(device)
                out_spikes_counter = net(img)
                correct_sum += (out_spikes_counter.argmax(dim=1) == label).float().sum().item()
                test_sum += label.numel()
                functional.reset_net(net)
            test_accuracy = correct_sum / test_sum
            print('test_accuracy', test_accuracy)
            log_data_list[-1].append(('test_accuracy', test_accuracy, net.epoch))
            if use_plif:
                plif_idx = 0
                for m in net.modules():
                    if isinstance(m, models.PLIFNode):
                        log_data_list[-1].append(('w' + str(plif_idx), m.w.item(), net.train_times))
                        plif_idx += 1

            print('Writing....')
            for item in log_data_list[-1]:
                writer.add_scalar(item[0], item[1], item[2])
            if net.max_test_accuracy <= test_accuracy:
                print('save model with test_accuracy = ', test_accuracy)
                net.max_test_accuracy = test_accuracy
                torch.save({
                    'net': net,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'log_data_list': log_data_list
                }, check_point_max_path)

            # 保存最新模型
            torch.save({
                'net': net,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'log_data_list': log_data_list
            }, check_point_path)

            if (time.time() - ckpt_time) / 3600 > 4:
                # 检查点
                # 每4小时保存一次
                torch.save({
                    'net': net,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'log_data_list': log_data_list
                }, os.path.join(pt_dir, f'check_point_{net.epoch}.pt'))

            print('Written.')

            speed_per_epoch = time.time() - start_time
            print('speed per epoch', speed_per_epoch)

    writer.close()
