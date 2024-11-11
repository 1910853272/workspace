# 使用SpikingJelly构建SNN对MNIST数据集进行识别

**spikingjellyMNIST.py**是该项目主要文件

**data**文件夹是MNIST数据集

**spikingjelly**文件夹是调用的SpikingJelly库

> spikingjelly\activation_based 文件夹下 neuron, encoding, functional, surrogate, layer，写实验报告时可以参考，定义了SNN神经元模型和编码方式等。

**log**文件夹下是对应参数`例如tau=2.0,T=100,batch_size=64,lr=1e-3`的两个npy文件、训练日志以及可视化结果。



终端输出结果如下：

![1](https://raw.githubusercontent.com/1910853272/image/master/img/202411111721434.png)

......

![2](https://raw.githubusercontent.com/1910853272/image/master/img/202411111719402.png)