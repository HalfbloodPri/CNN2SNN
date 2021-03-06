from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("CNN-Train/MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()
'''生成权值'''
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
'''卷积操作'''
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='VALID')
'''平均池化操作'''
def avg_pool_2x2(x):
    return tf.nn.avg_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

x = tf.placeholder(tf.float32, [None, 784])     #原始的n*784的向量
y_ = tf.placeholder(tf.float32, [None, 10])     #标签
x_image = tf.reshape(x, [-1,28,28,1])       #n*28*28的MNIST数据
#####训练过程
'''
第一层卷积，把28*28的图片分成12*12的9张小图片，每两张之间有4行/列像素点的重叠；
然后进行分组卷积，每张小图片进行独立的常规卷积操作，采用8个5*5的卷积核，每个
小图片得到8张大小为8*8的特征图，所以共输出8*9张8*8大小的特征图;
和第一层平均池化，池化核大小为2*2，得到8*9张4*4大小的特征图输出。
'''
W_conv1 = weight_variable([5,5,1,8])
h_conv1 = [None,None,None,None,None,None,None,None,None]
h_pool1 = [None,None,None,None,None,None,None,None,None]
for i in range(3):
    for j in range(3):
        h_conv1[i*3+j] = tf.nn.relu(conv2d(x_image[:,(8*i):(8*i+12),(8*j):(8*j+12),:],W_conv1))
        h_pool1[i*3+j] = avg_pool_2x2(h_conv1[i*3+j])
'''
将第一层池化得到的8*9张4*4大小的特征图拼成8张12*12的特征图;
'''
h_pool1_mergeOnDiv2 = [None,None,None]
for i in range(3):
    h_pool1_mergeOnDiv2[i] = tf.concat([h_pool1[3*i],h_pool1[3*i+1],h_pool1[3*i+2]],2)
h_pool1_merge = tf.concat([h_pool1_mergeOnDiv2[0],h_pool1_mergeOnDiv2[1],h_pool1_mergeOnDiv2[2]],1)
'''
第二层卷积和平均池化；
将12*12的特征图分成8*4张8*8的特征图，每张原始的12*12的特征图可以得到四张8*8的小特征图，每
两张小特征图会有四行或者四列特征点的重叠；采用32个5*5的卷积核，得到32*4张4*4大小的特征图。
池化之后得到32*4张2*2大小的特征图。
'''
W_conv2 = weight_variable([5,5,8,32])
h_conv2 = [None,None,None,None]
h_pool2 = [None,None,None,None]
for i in range(2):
    for j in range(2):
        h_conv2[i*2+j] = tf.nn.relu(conv2d(h_pool1_merge[:,(4*i):(4*i+8),(4*j):(4*j+8),:],W_conv2))
        h_pool2[i*2+j] = avg_pool_2x2(h_conv2[i*2+j])
'''
将第二层池化得到的32*4张2*2大小的特征图拼成32张4*4大小的特征图；
'''
h_pool2_mergeOnDiv2 = [None,None]
for i in range(2):
    h_pool2_mergeOnDiv2[i] = tf.concat([h_pool2[2*i],h_pool2[2*i+1]],2)
h_pool2_merge = tf.concat([h_pool2_mergeOnDiv2[0],h_pool2_mergeOnDiv2[1]],1)
'''
全连接层，将32张4*4大小的特征图，共32*4*4个点，与输出层的10个点相连,并经过softmax输出；
'''
W_fc = weight_variable([32*4*4,10])
h_pool2_flat = tf.reshape(h_pool2_merge,[-1,32*4*4])
y = tf.nn.softmax(tf.matmul(h_pool2_flat,W_fc))
'''定义损失值'''
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
Wloss_conv1 = tf.reduce_mean(tf.reduce_sum(tf.where(tf.greater(W_conv1,0.0), tf.square(W_conv1-1.0), tf.square(W_conv1+1.0)), reduction_indices=[1]))
Wloss_conv2 = tf.reduce_mean(tf.reduce_sum(tf.where(tf.greater(W_conv2,0.0), tf.square(W_conv2-1.0), tf.square(W_conv2+1.0)), reduction_indices=[1]))
Wloss_fc = tf.reduce_mean(tf.reduce_sum(tf.where(tf.greater(W_fc,0.0), tf.square(W_fc-1.0), tf.square(W_fc+1.0)), reduction_indices=[1]))
fullLoss = cross_entropy + 0.5*(Wloss_conv1+Wloss_conv2+Wloss_fc)
train_step = tf.train.AdamOptimizer(1e-5).minimize(fullLoss)
'''定义准确率'''
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
################################################################################
#####权值二值化之后的前向识别过程
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
################################################################################
#####定义Saver
saver = tf.train.Saver(tf.global_variables())
saver.restore(sess,"CNN-Train/mnistCNNBi-20000")
################################################################################
'''接下来是训练过程'''
#tf.global_variables_initializer().run()
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%1000 == 0:
        test_accuracy = accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels})
        print("step %d, test accuracy %g"%(i, test_accuracy))
    train_step.run(feed_dict={x: batch[0], y_:batch[1]})
'''验证在测试集上的准确率'''
print("test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
test_accuracy_Bi = accuracy_Bi.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels,\
                                    W_conv1_placeholder:sess.run(W_conv1), \
                                    W_conv2_placeholder:sess.run(W_conv2), \
                                    W_fc_placeholder:sess.run(W_fc)})
print("test accuracy_Bi %g"%test_accuracy_Bi)
'''存储模型和变量'''
saver.save(sess, 'CNN-Train/mnistCNNBi1_1', global_step=i+1)