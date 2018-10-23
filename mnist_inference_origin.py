import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

#第一层卷积层尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5

CONV2_DEEP = 64
CONV2_SIZE = 5

#全连接层的节点个数
FC_SIZE = 512

def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer = tf.truncated_normal_initializer(stddev = 0.1))
        conv1_biases = tf.get_variable(
            "bias", [CONV1_DEEP], initializer = tf.constant_initializer(0.0))

        #边长为5 深度为32的滤波器  滤波器步长为1 用全0填充
        conv1 = tf.nn.conv2d(
            input_tensor, conv1_weights, strides = [1,1,1,1], padding = 'SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))   #attention

    #实现第二层池化层的前向传播过程 池化层滤波器为2
    #使用全0填充且移动步长为2 这层输入为28x28x32 输出14x14x32
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(
            relu1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

    #实现第三层卷积层的变量以及前向传播 这层输入为14x14x32 
    #输出为14x14x64
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer = tf.truncated_normal_initializer(stddev = 0.1))
        conv2_biases = tf.get_variable(
            "bias", [CONV2_DEEP],
            initializer = tf.constant_initializer(0.0))

        #使用边长为5 深度为64的过滤器  过滤器移动步长为1 用0全填充
        conv2 = tf.nn.conv2d(
            pool1, conv2_weights, strides = [1,1,1,1], padding = 'SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))

    #实现第四层池化层的前向传播  这层和第二层结构一样 这层输入14x14x64矩阵  输出7x7x64
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(
            relu2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

    #将第四层的输出转为第五层全连接层的输入格式
    #pool_shape[0]为一个batch中数据个数
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

    #通过tf.reshape函数将第四层输出变成一个batch向量
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    #声明第五层前向传播 这一层输入时拉直后的一组向量
    #向量长度3136 输出512的向量 引入dropout概念，会在训练时随机将部分节点输出改为0
    #避免过拟合
    #dropout只在全连接层用

    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable(
            "weight",[nodes, FC_SIZE],
            initializer = tf.truncated_normal_initializer(stddev = 0.1))
        #只有全连接层需要加入正则化
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases = tf.get_variable(
            "bias", [FC_SIZE], initializer = tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train : fc1 = tf.nn.dropout(fc1, 0.5)

    #声明第六层全连接层的变量并实现前向传播 这层输入为512长度向量 输出长度为10
    #经过softmax后得到分类结果
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable(
            "weight", [FC_SIZE, NUM_LABELS],
            initializer = tf.truncated_normal_initializer(stddev = 0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable(
            "bias", [NUM_LABELS],
            initializer = tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

        return logit   