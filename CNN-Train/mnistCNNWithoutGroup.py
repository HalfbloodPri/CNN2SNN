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
Wloss_conv1 = tf.reduce_mean(tf.reduce_sum(tf.where(tf.greater(W_conv1,0.0), tf.square(W_conv1-1.0), tf.square(W_conv1+1.0)), reduction_indices=[1]))
Wloss_conv2 = tf.reduce_mean(tf.reduce_sum(tf.where(tf.greater(W_conv2,0.0), tf.square(W_conv2-1.0), tf.square(W_conv2+1.0)), reduction_indices=[1]))
Wloss_fc = tf.reduce_mean(tf.reduce_sum(tf.where(tf.greater(W_fc,0.0), tf.square(W_fc-1.0), tf.square(W_fc+1.0)), reduction_indices=[1]))
fullLoss = cross_entropy + 0.5*(Wloss_conv1+Wloss_conv2+Wloss_fc)
#train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(fullLoss)
train_step = tf.train.GradientDescentOptimizer(4e-4).minimize(fullLoss)
'''定义准确率'''
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#####定义Saver
saver = tf.train.Saver(tf.global_variables())
saver.restore(sess,"CNN-Train/mnistCNNFull3-40000")
################################################################################
'''接下来是训练过程'''
#tf.global_variables_initializer().run()
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%1000 == 0:
        #train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1]})
        #print("step %d, training accuracy %g"%(i, train_accuracy))
        print("step %d, test accuracy %g"%(i,accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels})))
    train_step.run(feed_dict={x: batch[0], y_:batch[1]})
'''验证在测试集上的准确率'''
print("test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels}))

saver.save(sess, 'CNN-Train/mnistCNNFull4', global_step=i+1)