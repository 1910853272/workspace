import os
import time
import argparse
import sys
import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
import torchvision

# 导入 SpikingJelly 库中的模块
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer

# 定义 SNN（Spiking Neural Network）模型类
class SNN(nn.Module):
    def __init__(self, tau):
        super().__init__()

        # 使用序列容器构建网络层
        self.layer = nn.Sequential(
            layer.Flatten(),  # 展平输入张量
            layer.Linear(28 * 28, 10, bias=False),  # 全连接层，将28x28像素展开为10个输出节点
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),  # 使用LIF神经元，设置时间常数tau和近似梯度函数
        )

    def forward(self, x: torch.Tensor):
        # 前向传播
        return self.layer(x)

# 可视化函数定义
def plot_2d_heatmap(array: np.ndarray, title: str, xlabel: str, ylabel: str, int_x_ticks=True, int_y_ticks=True,
                    plot_colorbar=True, colorbar_y_label='magnitude', x_max=None, figsize=(12, 8), dpi=200):
    if array.ndim != 2:
        raise ValueError(f"Expected 2D array, got {array.ndim}D array instead")

    fig, heatmap = plt.subplots(figsize=figsize, dpi=dpi)
    if x_max is not None:
        im = heatmap.imshow(array.T, aspect='auto', extent=[-0.5, x_max, array.shape[1] - 0.5, -0.5])
    else:
        im = heatmap.imshow(array.T, aspect='auto')

    heatmap.set_title(title)
    heatmap.set_xlabel(xlabel)
    heatmap.set_ylabel(ylabel)

    heatmap.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=int_x_ticks))
    heatmap.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=int_y_ticks))
    heatmap.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    heatmap.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    if plot_colorbar:
        cbar = heatmap.figure.colorbar(im)
        cbar.ax.set_ylabel(colorbar_y_label, rotation=90, va='top')
        cbar.ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    return fig

def plot_1d_spikes(spikes: np.ndarray, title: str, xlabel: str, ylabel: str, int_x_ticks=True, int_y_ticks=True,
                   plot_firing_rate=True, firing_rate_map_title='firing rate', figsize=(12, 8), dpi=200):
    if spikes.ndim != 2:
        raise ValueError(f"Expected 2D array, got {spikes.ndim}D array instead")

    spikes_T = spikes.T
    if plot_firing_rate:
        fig = plt.figure(tight_layout=True, figsize=figsize, dpi=dpi)
        gs = matplotlib.gridspec.GridSpec(1, 5)
        spikes_map = fig.add_subplot(gs[0, 0:4])
        firing_rate_map = fig.add_subplot(gs[0, 4])
    else:
        fig, spikes_map = plt.subplots()

    spikes_map.set_title(title)
    spikes_map.set_xlabel(xlabel)
    spikes_map.set_ylabel(ylabel)

    spikes_map.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=int_x_ticks))
    spikes_map.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=int_y_ticks))

    spikes_map.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    spikes_map.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    spikes_map.set_xlim(-0.5, spikes_T.shape[1] - 0.5)
    spikes_map.set_ylim(-0.5, spikes_T.shape[0] - 0.5)
    spikes_map.invert_yaxis()
    N = spikes_T.shape[0]
    T = spikes_T.shape[1]
    t = np.arange(0, T)
    t_spike = spikes_T * t
    mask = (spikes_T == 1)  # eventplot中的数值是时间发生的时刻，因此需要用mask筛选出

    colormap = plt.get_cmap('tab10')  # cmap的种类参见https://matplotlib.org/gallery/color/colormap_reference.html

    for i in range(N):
        spikes_map.eventplot(t_spike[i][mask[i]], lineoffsets=i, colors=colormap(i % 10))

    if plot_firing_rate:
        firing_rate = np.mean(spikes_T, axis=1, keepdims=True)

        max_rate = firing_rate.max()
        min_rate = firing_rate.min()

        firing_rate_map.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        firing_rate_map.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        firing_rate_map.imshow(firing_rate, cmap='magma', aspect='auto')
        for i in range(firing_rate.shape[0]):
            firing_rate_map.text(0, i, f'{firing_rate[i][0]:.2f}', ha='center', va='center', color='w' if firing_rate[i][0] < 0.7 * max_rate or min_rate == max_rate else 'black')
        firing_rate_map.get_xaxis().set_visible(False)
        firing_rate_map.set_title(firing_rate_map_title)
    return fig

def plot_one_neuron_v_s(v: np.ndarray, s: np.ndarray, v_threshold=1.0, v_reset=0.0,
                        title='$V[t]$ and $S[t]$ of the neuron', figsize=(12, 8), dpi=200):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax0.set_title(title)
    T = s.shape[0]
    t = np.arange(0, T)
    ax0.plot(t, v)
    ax0.set_xlim(-0.5, T - 0.5)
    ax0.set_ylabel('Voltage')
    ax0.axhline(v_threshold, label='$V_{threshold}$', linestyle='-.', c='r')
    if v_reset is not None:
        ax0.axhline(v_reset, label='$V_{reset}$', linestyle='-.', c='g')
    ax0.legend(frameon=True)
    t_spike = s * t
    mask = (s == 1)  # eventplot中的数值是时间发生的时刻，因此需要用mask筛选出
    ax1 = plt.subplot2grid((3, 1), (2, 0))
    ax1.eventplot(t_spike[mask], lineoffsets=0, colors='r')
    ax1.set_xlim(-0.5, T - 0.5)

    ax1.set_xlabel('Simulating Step')
    ax1.set_ylabel('Spike')
    ax1.set_yticks([])

    ax1.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    return fig, ax0, ax1

def main():
    '''
    使用全连接-LIF的网络结构，进行MNIST识别。
    这个函数会初始化网络进行训练，并显示训练过程中在测试集的正确率。
    '''

    # 命令行参数解析
    parser = argparse.ArgumentParser(description='LIF MNIST Training')
    parser.add_argument('-T', default=100, type=int, help='simulating time-steps')  # 仿真时间步数
    parser.add_argument('-device', default='cuda:0', help='device')  # 设备，默认为CUDA设备0
    parser.add_argument('-b', default=64, type=int, help='batch size')  # 批量大小
    parser.add_argument('-epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')  # 训练的总轮数
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')  # 数据加载的工作线程数
    parser.add_argument('-data-dir', type=str, default='./data', help='root dir of MNIST dataset')  # MNIST数据集的根目录
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')  # 日志和检查点的保存目录
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')  # 从检查点路径恢复训练
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')  # 是否使用自动混合精度训练
    parser.add_argument('-opt', type=str, choices=['sgd', 'adam'], default='adam', help='use which optimizer. SGD or Adam')  # 选择优化器
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')  # SGD优化器的动量参数
    parser.add_argument('-lr', default=1e-3, type=float, help='learning rate')  # 学习率
    parser.add_argument('-tau', default=2.0, type=float, help='parameter tau of LIF neuron')  # LIF神经元的时间常数tau

    args = parser.parse_args()
    print(args)

    # 初始化网络模型
    net = SNN(tau=args.tau)
    print(net)
    net.to(args.device)  # 将模型移动到指定设备

    # 初始化数据集和数据加载器
    train_dataset = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )

    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )
    test_data_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=args.b,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )

    # 混合精度训练的缩放器
    scaler = None
    if args.amp:
        scaler = amp.GradScaler()  # 初始化混合精度缩放器

    start_epoch = 0
    max_test_acc = -1  # 记录最高测试准确率

    # 初始化优化器
    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    # 如果提供了检查点路径，则加载检查点
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']

    # 设置输出目录
    out_dir = os.path.join(args.out_dir, f'T{args.T}_b{args.b}_{args.opt}_lr{args.lr}')
    if args.amp:
        out_dir += '_amp'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    # 保存命令行参数到文本文件
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))

    # 初始化TensorBoard日志记录器
    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))

    # 初始化编码器（泊松编码器）
    encoder = encoding.PoissonEncoder()

    # Initialize lists to store accuracies
    train_acc_list = []
    test_acc_list = []
    epoch_list = []

    # 开始训练循环
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()  # 设置模型为训练模式
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in train_data_loader:
            optimizer.zero_grad()  # 清零梯度
            img = img.to(args.device)
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 10).float()  # 将标签转换为one-hot编码

            if scaler is not None:
                with amp.autocast():
                    out_fr = 0.
                    for t in range(args.T):
                        encoded_img = encoder(img)  # 对图像进行编码
                        out_fr += net(encoded_img)  # 前向传播
                    out_fr = out_fr / args.T  # 计算平均发放率
                    loss = F.mse_loss(out_fr, label_onehot)  # 计算损失
                scaler.scale(loss).backward()  # 反向传播
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = 0.
                for t in range(args.T):
                    encoded_img = encoder(img)
                    out_fr += net(encoded_img)
                out_fr = out_fr / args.T
                loss = F.mse_loss(out_fr, label_onehot)
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)  # 重置网络状态

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        # 记录训练损失和准确率
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        train_acc_list.append(train_acc)  # Append train accuracy
        epoch_list.append(epoch)  # Append current epoch

        # 开始测试
        net.eval()  # 设置模型为评估模式
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(args.device)
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 10).float()
                out_fr = 0.
                for t in range(args.T):
                    encoded_img = encoder(img)
                    out_fr += net(encoded_img)
                out_fr = out_fr / args.T
                loss = F.mse_loss(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)
        test_acc_list.append(test_acc)  # Append test accuracy

        # 检查是否达到新的最高测试准确率
        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        # 保存检查点
        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        # 打印训练和测试信息
        print(args)
        print(out_dir)
        print(f'epoch ={epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')

    # 绘制训练和测试准确率曲线
    plt.figure()
    plt.plot(epoch_list, train_acc_list, label='Train Accuracy')
    plt.plot(epoch_list, test_acc_list, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Train and Test Accuracy over Epochs')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'accuracy_curve.png'))
    plt.close()

    # 保存并可视化用于绘图的数据
    net.eval()
    # 注册钩子函数以保存神经元状态
    output_layer = net.layer[-1]  # 获取输出层
    output_layer.v_seq = []
    output_layer.s_seq = []
    def save_hook(m, x, y):
        m.v_seq.append(m.v.unsqueeze(0))
        m.s_seq.append(y.unsqueeze(0))

    output_layer.register_forward_hook(save_hook)

    # 对测试集的第一个样本进行推理并保存状态
    with torch.no_grad():
        img, label = test_dataset[0]
        img = img.unsqueeze(0)  # 增加batch维度
        img = img.to(args.device)
        out_fr = 0.
        for t in range(args.T):
            encoded_img = encoder(img)
            out_fr += net(encoded_img)
        out_spikes_counter_frequency = (out_fr / args.T).cpu().numpy()
        print(f'Firing rate: {out_spikes_counter_frequency}')

        output_layer.v_seq = torch.cat(output_layer.v_seq)
        output_layer.s_seq = torch.cat(output_layer.s_seq)
        v_t_array = output_layer.v_seq.cpu().numpy().squeeze()  # v_t_array[i][j]表示神经元i在时刻j的膜电位
        s_t_array = output_layer.s_seq.cpu().numpy().squeeze()  # s_t_array[i][j]表示神经元i在时刻j的脉冲输出，值为0或1

        # 保存数据
        np.save(os.path.join(out_dir, "v_t_array.npy"), v_t_array)
        np.save(os.path.join(out_dir, "s_t_array.npy"), s_t_array)

        # 可视化电压和脉冲
        # 绘制所有输出神经元的电压热力图
        fig = plot_2d_heatmap(
            array=v_t_array,
            title='Output Layer Membrane Potentials',
            xlabel='Simulating Step',
            ylabel='Neuron Index',
            int_x_ticks=True,
            x_max=args.T,
            dpi=200
        )
        plt.savefig(os.path.join(out_dir, 'output_layer_v_heatmap.png'))
        plt.close(fig)

        # 绘制所有输出神经元的脉冲发放情况
        fig = plot_1d_spikes(
            spikes=s_t_array,
            title='Output Layer Spikes',
            xlabel='Simulating Step',
            ylabel='Neuron Index',
            dpi=200
        )
        plt.savefig(os.path.join(out_dir, 'output_layer_spikes.png'))
        plt.close(fig)

        # 绘制单个神经元的电压和脉冲（以第0个神经元为例）
        neuron_index = 0
        v_neuron = v_t_array[:, neuron_index]
        s_neuron = s_t_array[:, neuron_index]
        fig, ax0, ax1 = plot_one_neuron_v_s(
            v=v_neuron,
            s=s_neuron,
            v_threshold=output_layer.v_threshold,
            v_reset=output_layer.v_reset,
            title=f'Neuron {neuron_index} Voltage and Spikes',
            dpi=200
        )
        plt.savefig(os.path.join(out_dir, f'neuron_{neuron_index}_v_s.png'))
        plt.close(fig)

if __name__ == '__main__':
    main()
