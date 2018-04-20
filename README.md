# CNN2SNN
参考文献：Spike Trains Encoding and Threshold Rescaling Method for Deep Spiking Neural Networks；
按照参考文献的方法训练CNN，并尝试转换为SNN，采用发射率编码；

##训练CNN时的约束：
1、激活函数采用ReLU；
2、偏置全为0；
3、采用平均池化；

##文件夹
./CNN-Train : 按照约束条件进行CNN训练，目标数据集是MNIST；