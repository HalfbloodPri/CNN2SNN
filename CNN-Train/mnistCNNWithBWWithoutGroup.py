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
第一层卷积，采用8个5*5的卷积核，输出8张24*24大小的特征图;
和第一层平均池化，池化核大小为2*2，得到8张12*12大小的特征图输出。
'''
W_conv1 = weight_variable([5,5,1,8])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1))
h_pool1 = avg_pool_2x2(h_conv1)
'''
第二层卷积和平均池化；
采用32个5*5的卷积核，得到32张8*8大小的特征图。
池化之后得到32张4*4大小的特征图。
'''
W_conv2 = weight_variable([5,5,8,32])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2))
h_pool2 = avg_pool_2x2(h_conv2)
'''
全连接层，将32张4*4大小的特征图，共32*4*4个点，与输出层的10个点相连,并经过softmax输出；
'''
W_fc = weight_variable([32*4*4,10])
h_pool2_flat = tf.reshape(h_pool2,[-1,32*4*4])
y = tf.nn.softmax(tf.matmul(h_pool2_flat,W_fc))
'''定义损失值'''
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
Wloss_conv1 = 0.2*tf.reduce_mean(tf.reduce_sum(tf.where(tf.greater(W_conv1,0.0), tf.square(W_conv1-1.0), tf.square(W_conv1+1.0)), reduction_indices=[1]))
Wloss_conv2 = 0.2*0.125*tf.reduce_mean(tf.reduce_sum(tf.where(tf.greater(W_conv2,0.0), tf.square(W_conv2-1.0), tf.square(W_conv2+1.0)), reduction_indices=[1]))
Wloss_fc = 0.2*0.0625*tf.reduce_mean(tf.reduce_sum(tf.where(tf.greater(W_fc,0.0), tf.square(W_fc-1.0), tf.square(W_fc+1.0)), reduction_indices=[1]))
fullLoss = cross_entropy + Wloss_conv1 + Wloss_conv2 + Wloss_fc
train_step = tf.train.GradientDescentOptimizer(1e-1).minimize(fullLoss)
'''定义准确率'''
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
################################################################################
#####权值二值化之后的前向识别过程
W_conv1_placeholder = tf.placeholder(tf.float32, [5,5,1,8])
W_conv2_placeholder = tf.placeholder(tf.float32, [5,5,8,32])
W_fc_placeholder = tf.placeholder(tf.float32, [32*4*4,10])
'''
第一层卷积，采用8个5*5的卷积核，输出8张24*24大小的特征图;
和第一层平均池化，池化核大小为2*2，得到8张12*12大小的特征图输出。
'''
W1mat1 = tf.ones((5,5,1,8))
W1matSub1 = -tf.ones((5,5,1,8))
W_conv1_Bi = tf.where(W_conv1_placeholder<0,W1matSub1,W1mat1)
h_conv1_Bi = tf.nn.relu(conv2d(x_image,W_conv1_Bi))
h_pool1_Bi = avg_pool_2x2(h_conv1_Bi)
'''
第二层卷积和平均池化；
采用32个5*5的卷积核，得到32张8*8大小的特征图。
池化之后得到32张4*4大小的特征图。
'''
W2mat1 = tf.ones((5,5,8,32))
W2matSub1 = -tf.ones((5,5,8,32))
W_conv2_Bi = tf.where(W_conv2_placeholder<0,W2matSub1,W2mat1)
h_conv2_Bi = tf.nn.relu(conv2d(h_pool1_Bi,W_conv2_Bi))
h_pool2_Bi = avg_pool_2x2(h_conv2_Bi)
'''
全连接层，将32张4*4大小的特征图，共32*4*4个点，与输出层的10个点相连,并经过softmax输出；
'''
fcMat1 = tf.ones((32*4*4,10))
fcMatSub1 = -tf.ones((32*4*4,10))
W_fc_Bi = tf.where(W_fc_placeholder<0,fcMatSub1,fcMat1)
h_pool2_flat_Bi = tf.reshape(h_pool2_Bi,[-1,32*4*4])
y_Bi = tf.nn.softmax(tf.matmul(h_pool2_flat_Bi,W_fc_Bi))
'''定义准确率'''
correct_prediction_Bi = tf.equal(tf.argmax(y_Bi,1),tf.argmax(y_,1))
accuracy_Bi = tf.reduce_mean(tf.cast(correct_prediction_Bi, tf.float32))
################################################################################
#####定义Saver
saver = tf.train.Saver(tf.global_variables())
#saver.restore(sess,"CNN-Train/mnistCNNBiGDO1-20000")
################################################################################
'''接下来是训练过程'''
tf.global_variables_initializer().run()
for i in range(10000):
    batch = mnist.train.next_batch(50)
    if i%1000 == 0:
        test_accuracy = accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels})
        print("step %d, test accuracy %g"%(i, test_accuracy))
        print(sess.run(cross_entropy,feed_dict={x:mnist.test.images, y_:mnist.test.labels}),\
              sess.run(Wloss_conv1),sess.run(Wloss_conv2),sess.run(Wloss_fc))
    train_step.run(feed_dict={x: batch[0], y_:batch[1]})
'''验证在测试集上的准确率'''
print("test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
test_accuracy_Bi = accuracy_Bi.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels,\
                                    W_conv1_placeholder:sess.run(W_conv1), \
                                    W_conv2_placeholder:sess.run(W_conv2), \
                                    W_fc_placeholder:sess.run(W_fc)})
print("test accuracy_Bi %g"%test_accuracy_Bi)
print(sess.run(W_fc))
'''存储模型和变量'''
saver.save(sess, 'CNN-Train/mnistCNNBiGDO0', global_step=i+1)