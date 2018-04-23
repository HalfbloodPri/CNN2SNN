from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
'''
在前向计算的过程中，采用二值化的权重验证识别率，大于等于0的权重为1，小于0的权重为-1；
'''
mnist = input_data.read_data_sets("CNN-Train/MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()
'''卷积操作'''
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='VALID')
'''平均池化操作'''
def avg_pool_2x2(x):
    return tf.nn.avg_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

x = tf.placeholder(tf.float32, [None, 784])     #原始的n*784的向量
y_ = tf.placeholder(tf.float32, [None, 10])     #标签
x_image = tf.reshape(x, [-1,28,28,1])       #n*28*28的MNIST数据

W_conv1_placeholder = tf.placeholder(tf.float32, [5,5,1,8])
W_conv2_placeholder = tf.placeholder(tf.float32, [5,5,8,32])
W_fc_placeholder = tf.placeholder(tf.float32, [32*4*4,10])

'''
第一层卷积，把28*28的图片分成12*12的9张小图片，每两张之间有4行/列像素点的重叠；
然后进行分组卷积，每张小图片进行独立的常规卷积操作，采用8个5*5的卷积核，每个
小图片得到8张大小为8*8的特征图，所以共输出8*9张8*8大小的特征图;
和第一层平均池化，池化核大小为2*2，得到8*9张4*4大小的特征图输出。
'''
W1mat1 = tf.ones((5,5,1,8))
W1matSub1 = -tf.ones((5,5,1,8))
W_conv1_Bi = tf.where(W_conv1_placeholder<0,W1matSub1,W1mat1)
h_conv1_Bi = [None,None,None,None,None,None,None,None,None]
h_pool1_Bi = [None,None,None,None,None,None,None,None,None]
for i in range(3):
    for j in range(3):
        h_conv1_Bi[i*3+j] = tf.nn.relu(conv2d(x_image[:,(8*i):(8*i+12),(8*j):(8*j+12),:],W_conv1_Bi))
        h_pool1_Bi[i*3+j] = avg_pool_2x2(h_conv1_Bi[i*3+j])
'''
将第一层池化得到的8*9张4*4大小的特征图拼成8张12*12的特征图;
'''
h_pool1_mergeOnDiv2_Bi = [None,None,None]
for i in range(3):
    h_pool1_mergeOnDiv2_Bi[i] = tf.concat([h_pool1_Bi[3*i],h_pool1_Bi[3*i+1],h_pool1_Bi[3*i+2]],2)
h_pool1_merge_Bi = tf.concat([h_pool1_mergeOnDiv2_Bi[0],h_pool1_mergeOnDiv2_Bi[1],h_pool1_mergeOnDiv2_Bi[2]],1)
'''
第二层卷积和平均池化；
将12*12的特征图分成8*4张8*8的特征图，每张原始的12*12的特征图可以得到四张8*8的小特征图，每
两张小特征图会有四行或者四列特征点的重叠；采用32个5*5的卷积核，得到32*4张4*4大小的特征图。
池化之后得到32*4张2*2大小的特征图。
'''
W2mat1 = tf.ones((5,5,8,32))
W2matSub1 = -tf.ones((5,5,8,32))
W_conv2_Bi = tf.where(W_conv2_placeholder<0,W2matSub1,W2mat1)
h_conv2_Bi = [None,None,None,None]
h_pool2_Bi = [None,None,None,None]
for i in range(2):
    for j in range(2):
        h_conv2_Bi[i*2+j] = tf.nn.relu(conv2d(h_pool1_merge_Bi[:,(4*i):(4*i+8),(4*j):(4*j+8),:],W_conv2_Bi))
        h_pool2_Bi[i*2+j] = avg_pool_2x2(h_conv2_Bi[i*2+j])
'''
将第二层池化得到的32*4张2*2大小的特征图拼成32张4*4大小的特征图；
'''
h_pool2_mergeOnDiv2_Bi = [None,None]
for i in range(2):
    h_pool2_mergeOnDiv2_Bi[i] = tf.concat([h_pool2_Bi[2*i],h_pool2_Bi[2*i+1]],2)
h_pool2_merge_Bi = tf.concat([h_pool2_mergeOnDiv2_Bi[0],h_pool2_mergeOnDiv2_Bi[1]],1)
'''
全连接层，将32张4*4大小的特征图，共32*4*4个点，与输出层的10个点相连,并经过softmax输出；
'''
fcMat1 = tf.ones((32*4*4,10))
fcMatSub1 = -tf.ones((32*4*4,10))
W_fc_Bi = tf.where(W_fc_placeholder<0,fcMatSub1,fcMat1)
h_pool2_flat_Bi = tf.reshape(h_pool2_merge_Bi,[-1,32*4*4])
y_Bi = tf.nn.softmax(tf.matmul(h_pool2_flat_Bi,W_fc_Bi))
'''定义准确率'''
correct_prediction_Bi = tf.equal(tf.argmax(y_Bi,1),tf.argmax(y_,1))
accuracy_Bi = tf.reduce_mean(tf.cast(correct_prediction_Bi, tf.float32))