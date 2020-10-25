# -*- coding:utf-8 -*-
'''
乔家阳
时间:2020年9月20日
单位:上海交通大学&Qiyang education technology group
描述:该文件主要描述的是关于使用LSTM的脚本文件，解决的任务是文本二分类。在该脚本当中，我们首先对文本进行预处理:[文本数字化]、[文本数字
数据修整]、[文本数字数据的词向量扩充]。
接下来是对LSTM计算流图的建立，最后是将数据注入到计算流图当中进行实际的参数优化。
'''
import math
import re
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from Parameters_Enum import parameters as Pa
# 数据的读取
train = pd.read_csv(Pa.TRAIN_FILE, sep='\t')
test = pd.read_csv(Pa.TEST_FILE, sep='\t')
train_data = train.loc[:, 'data']
train_labels = train.loc[:, 'labels']
test_data = test.loc[:, 'data']
test_labels = test.loc[:, 'labels']

# lstm 需要的参数
def _parameters():
    return tf.contrib.training.HParams(
        num_embedding_size=16,  # [文本数字数据的词向量扩充]步骤当中，将单词数字扩充成的词向量的维度。
        num_timesteps=50,  # 该变量即代表了lstm模块当中的时间序列长度，又表示了我们所修整的一个文本当中的单词个数。
        vocab_size=6000,#在文本数字化的过程当中，单词向数字进行映射的时候，我们所进行映射的单词种类数。超过6000以外，对应0值
        num_lstm_nodes=32,  # lstm单元的参数的维度，如h、c的维度。h:[1,num_lstm_nodes]
        num_lstm_layers=1,  # lstm模块的层数
        num_fc_nodes=32,  # lstm在处理完一句文本之后，会将输出的信息h再投入到分类模型当中进行分类。该脚本使用的是逻辑回归。
        batch_size=500,#我们每进行一次epoch的训练，所选取的文本的数目，每一个文本都是进行并行的计算。
        clip_lstm_grads=1.0,#梯度截断的数值，在计算梯度值的时候，超过1.0的梯度绝对值，统一处理成1.0.
        learning_rate=0.001,#学习速率，小的学习速率可保证梯度稳定地进行下降。
    )

hps = _parameters()  #我们通过该对象来获取我们所设定的参数，如hps.vocab_size是来获取我们所设定的处理的单词种类数

# 随机抽选数据的函数
def random_select_batch(batch_size, train):
    row_rand_array = np.arange(train.shape[0])#我们以数据的标号来组成一个np数组,如[1,2,3]
    np.random.shuffle(row_rand_array)#我们对标号np数组当中的标号进行洗牌,如[2,3,1]
    batch_ = train[row_rand_array[0:batch_size]]#我们挑取洗牌后np数组的前batch_size个标号，从train数据集当中抽取出来
    batch_labels = batch_[:, 0]#将抽取出的数据集的标号单独抽取出来
    batch_labels = batch_labels.reshape(-1, 1)#将label的维度从(n,)修改成(n,1)，目的是要符合计算流图输入数据的要求
    batch_data = batch_[:, 1:]#获取抽取的batch数据的数据部分。
    return batch_data, batch_labels


# 数据的预处理，将单词全部改为小写的格式。这个函数是我从网上搜的。单词的大小写并没有蕴含什么有价值的信息。
def clear_review(text):
    texts = []
    for item in text:
        item = item.replace("<br /><br />", "")
        item = re.sub("[^a-zA-Z]", " ", item.lower())
        texts.append(" ".join(item.split()))
    return texts


# 删除停用词+词形还原，停用词所具备的价值信息极少量。我们为了缩减文本数据的大小，进行停用词处理。而英文单词的不同状态也不具有
#显著的价值信息，因此我们将单词的各种时态进行还原，都只使用原型。
def stemed_words(text):
    stop_words = stopwords.words("english")
    lemma = WordNetLemmatizer()
    texts = []
    for item in text:
        words = [lemma.lemmatize(w, pos='v') for w in item.split() if w not in stop_words]
        texts.append(" ".join(words))
    return texts


# 文本预处理
def preprocess(text):
    text = clear_review(text)
    text = stemed_words(text)
    return text


train_data_prc = preprocess(train_data)#将训练数据的纯数据部分进行文本预处理工作。
t = Tokenizer(num_words=hps.vocab_size)#t负责将单词与数字对应起来，完成文本的数字化处理工作。
t.fit_on_texts(train_data_prc)#对t对训练数据进行分析，t内部完成单词与数字直接的匹配。
new_train_data = t.texts_to_sequences(train_data_prc)  #将训练集的文本数据处理成数字数据。
new_train_sequence = keras.preprocessing.sequence.pad_sequences(
    new_train_data, maxlen=hps.num_timesteps,
    padding='post', truncating='post')#我们对文本的长度进行统一化控制，因为lstm的时间步长是固定的，文本的长度为num_timesteps
test_data_prc = preprocess(test_data)#对测试文本数据集进行预处理工作
new_test_data = t.texts_to_sequences(test_data_prc)#按照与训练数据相同的对应方式来将文本数据转化为数字数据
new_test_sequence = keras.preprocessing.sequence.pad_sequences(
    new_test_data, maxlen=hps.num_timesteps
    , padding='post', truncating='post') #我们对测试数据进行修正，保证数据能够符合lstm模块的时间步长
#new_test_sequence = new_test_sequence.reshape(-1, hps.num_timesteps)
np_train_labels = np.array(train_labels)#将训练集的标签
np_train_labels = np_train_labels.reshape(-1, 1)
new_test_labels = np.array(test_labels)
new_test_labels = new_test_labels.reshape(-1, 1)
new_train = np.concatenate((np_train_labels, new_train_sequence), axis=1)
new_test = np.concatenate((new_test_labels, new_test_sequence), axis=1)


# 开始计算图模型 （重点）
def create_model(hps, vocab_size, num_classes):
    '''
    构建lstm
    :param hps: 参数对象
    :param vocab_size:  词表 长度
    :param num_classes:  分类数目
    :return:
    '''
    num_timesteps = hps.num_timesteps  # 一个句子中 有 num_timesteps 个词语
    batch_size = hps.batch_size

    # 设置两个 placeholder， 内容id 和 标签id
    inputs = tf.placeholder(tf.int32, (batch_size, num_timesteps))
    outputs = tf.placeholder(tf.int32, (batch_size,))

    # dropout keep_prob 表示要keep多少值，丢掉的是1-keep_prob
    # keep_prob = tf.placeholder(tf.float32,name='keep_prob')

    global_step = tf.Variable(
        tf.zeros([], tf.int64),
        name='global_step',
        trainable=False)  # 可以保存 当前训练到了 哪一步，而且不训练

    # 随机的在均匀分布下初始化, 构建 embeding 层
    embeding_initializer = tf.random_uniform_initializer(-1.0, 1.0)

    # 和 name_scope 作用是一样的，他可以定义指定 initializer
    # tf.name_scope() 和 tf.variable_scope() 的区别 参考：
    # https://www.cnblogs.com/adong7639/p/8136273.html
    with tf.variable_scope('embedding', initializer=embeding_initializer):
        # tf.varialble_scope() 一般 和 tf.get_variable() 进行配合
        # 构建一个 embedding 矩阵,shape 是 [词表的长度, 每个词的embeding长度 ]
        embeddings = tf.get_variable('embedding', [vocab_size, hps.num_embedding_size], tf.float32)

        # 每一个词，都要去embedding中查找自己的向量
        # [1, 10, 7] 是一个句子，根据 embedding 进行转化
        # 如： [1, 10, 7] -> [embedding[1], embedding[10], embedding[7]]
        embeding_inputs = tf.nn.embedding_lookup(embeddings, inputs)
        # 上句的输入： Tensor("embedding/embedding_lookup:0", shape=(100, 50, 16), dtype=float32)
        # 输出是一个三维矩阵，分别是：100 是 batch_size 大小，50 是 句子中的单词数量，16 为 embedding 向量长度

    # lstm 层

    # 输入层 大小 加上 输出层的大小，然后开方
    scale = 1.0 / math.sqrt(hps.num_embedding_size + hps.num_lstm_nodes) / 3.0
    lstm_init = tf.random_uniform_initializer(-scale, scale)

    def _generate_parames_for_lstm_cell(x_size, h_size, bias_size):
        '''
        生成参数的变量
        :param x_size: x × w  其中 w 的形状
        :param h_size: 上一层 输出h 的形状
        :param bias_size: 偏置的形状
        :return: 各个 变量
        '''
        x_w = tf.get_variable('x_weights', x_size)  # 输入x的w权重的值
        h_w = tf.get_variable('h_weights', h_size)  # 上一层 输出h 的 值
        b = tf.get_variable('biases', bias_size, initializer=tf.constant_initializer(0.0))  # 偏置的 值

        return x_w, h_w, b

    with tf.variable_scope('lstm_nn', initializer=lstm_init):
        # 生成 四组 可变 参数，分别是 遗忘门、输入门、输出门  和 tanh
        # 输入门
        with tf.variable_scope('inputs'):
            ix, ih, ib = _generate_parames_for_lstm_cell(  # 以i开头，代表 inputs
                x_size=[hps.num_embedding_size, hps.num_lstm_nodes],  # []
                h_size=[hps.num_lstm_nodes, hps.num_lstm_nodes],
                bias_size=[1, hps.num_lstm_nodes]
            )
        with tf.variable_scope('outputs'):
            ox, oh, ob = _generate_parames_for_lstm_cell(  # 以i开头，代表 inputs
                x_size=[hps.num_embedding_size, hps.num_lstm_nodes],  # []
                h_size=[hps.num_lstm_nodes, hps.num_lstm_nodes],
                bias_size=[1, hps.num_lstm_nodes]
            )

        with tf.variable_scope('forget'):
            fx, fh, fb = _generate_parames_for_lstm_cell(  # 以i开头，代表 inputs
                x_size=[hps.num_embedding_size, hps.num_lstm_nodes],  # []
                h_size=[hps.num_lstm_nodes, hps.num_lstm_nodes],
                bias_size=[1, hps.num_lstm_nodes]
            )
        with tf.variable_scope('memory'):
            cx, ch, cb = _generate_parames_for_lstm_cell(  # 以i开头，代表 inputs
                x_size=[hps.num_embedding_size, hps.num_lstm_nodes],  # []
                h_size=[hps.num_lstm_nodes, hps.num_lstm_nodes],
                bias_size=[1, hps.num_lstm_nodes]
            )

        # 初始化 隐状态 隐状态的形状 (batch_size, lstm最后一层神经个数)
        state = tf.Variable(tf.zeros([batch_size, hps.num_lstm_nodes]),
                            trainable=False
                            )

        # 每个神经元的输出 形状同上
        h = tf.Variable(tf.zeros([batch_size, hps.num_lstm_nodes]),
                        trainable=False
                        )
        #该tensor将用于收集lstm模块的每一步的输出结果
        lstm_outputs = tf.Variable(tf.zeros([batch_size,hps.num_lstm_nodes,1]),
                                   trainable=False
                                   )
        for i in range(num_timesteps):  # 按照 词语的数量 进行
            # [batch_size, 1, embed_size]
            embeding_input = embeding_inputs[:, i, :]  # 取出 句子中 的 第一个词语(当i为1时)
            # 这样每次取出来的 中间的那一维度 就是 1，可以将其合并掉

            # 因为是 只有一个词语，所以将其reshape成 二维
            embeding_input = tf.reshape(embeding_input, [batch_size, hps.num_embedding_size])

            # 遗忘门
            forget_gate = tf.sigmoid(
                # 输入x与w相乘，加上 上一层输出h与hw相乘，在加上，偏置
                # 以下各个门同理
                tf.matmul(embeding_input, fx) + tf.matmul(h, fh) + fb
            )

            # 输入门
            input_gate = tf.sigmoid(
                tf.matmul(embeding_input, ix) + tf.matmul(h, ih) + ib
            )

            # 输出门
            output_gate = tf.sigmoid(
                tf.matmul(embeding_input, ox) + tf.matmul(h, oh) + ob
            )

            # tanh 层
            mid_state = tf.tanh(
                tf.matmul(embeding_input, cx) + tf.matmul(h, ch) + cb
            )

            # c状态 是 上一个单元传入c状态×遗忘门 再加上 输入门×tanh
            state = mid_state * input_gate + state * forget_gate

            h = output_gate * tf.tanh(state)
            lstm_outputs = tf.concat([lstm_outputs,tf.reshape(h,[batch_size, hps.num_lstm_nodes,1])],axis=2)
            s=1
        #last = h  # 只需要 最后一个 输出 就可以了
        lstm_outputs = lstm_outputs[:,:,1:]
        s=1

        #对lstm_outputs的各个时间步进行加权的权重系数矩阵
        with tf.variable_scope('time_weights',initializer=tf.constant_initializer(0.02)):
            weights_outputs = tf.get_variable('weights',[hps.num_timesteps,1])
            b_outputs = tf.get_variable('bias',[1,hps.num_lstm_nodes],initializer=tf.constant_initializer(0.0))
        m = tf.reshape(lstm_outputs,[hps.batch_size*hps.num_lstm_nodes,hps.num_timesteps])
        weighted_outputs_ =tf.matmul(m,weights_outputs)#16000*1
        weighted_outputs_2 = tf.reshape(weighted_outputs_,[hps.batch_size,hps.num_lstm_nodes])
        weighted_outputs = tf.nn.tanh(weighted_outputs_2+b_outputs)

        # 输出 Tensor("lstm_nn/mul_149:0", shape=(100, 32), dtype=float32)
        # 和注释部分的 last 输出 是同样的结果

    # 将最后一层的输出 链接到一个全连接层上
    # 参考链接：https://www.w3cschool.cn/tensorflow_python/tensorflow_python-fy6t2o0o.html
    fc_init = tf.uniform_unit_scaling_initializer(factor=1.0)
    with tf.variable_scope('fc', initializer=fc_init):  # initializer 此范围内变量的默认初始值
        fc1 = tf.layers.dense(weighted_outputs,
                              hps.num_fc_nodes,
                              activation=tf.nn.relu,
                              name='fc1')
        # 进行 dropout
        fc1_dropout = tf.nn.dropout(fc1, 0.8)
        # 进行更换 参考：https://blog.csdn.net/UESTC_V/article/details/79121642

        logits = tf.layers.dense(fc1_dropout, 1, name='fc2')
        logits = tf.reshape(logits, (logits.shape[0],))
        outputs = tf.cast(outputs, tf.float32)
    # 没有东西需要初始化，所以可以直接只用name_scope()
    with tf.name_scope('metrics'):
        binary_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits,
            labels=outputs
        )
        '''
        softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = logits,
            labels = outputs
        )
        '''
        # 该方法 做了三件事：1,labels 做 onehot，logits 计算softmax概率，3. 做交叉熵
        loss = tf.reduce_mean(binary_loss)
        y_pred = tf.round(
            tf.nn.sigmoid(logits)
        )
        '''
        y_pred = tf.argmax(
            tf.nn.softmax(logits),
            1,
            #output_type = tf.int64
        )
        '''

        # 这里做了 巨大 修改，如果问题，优先检查这里！！！！！！
        # print(type(outputs), type(y_pred))
        correct_pred = tf.equal(tf.cast(outputs, tf.int32), tf.cast(y_pred, tf.int32))  # 这里也做了修改
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.name_scope('train_op'):
        tvars = tf.trainable_variables()  # 获取所有可以训练的变量
        for var in tvars:
            tf.logging.info('variable name: %s' % (var.name))  # 打印出所有可训练变量

        # 对 梯度进行 截断.
        # grads是截断之后的梯度
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(loss, tvars),  # 在可训练的变量的梯度
            hps.clip_lstm_grads
        )  # 可以 获得 截断后的梯度
        grad_input = tf.gradients(loss, embeding_inputs)
        optimizer = tf.train.AdamOptimizer(hps.learning_rate)  # 将每个梯度应用到每个变量上去
        train_op = optimizer.apply_gradients(
            zip(grads, tvars),  # 将 梯度和参数 绑定起来
            global_step=global_step  # 这个参数 等会儿，再好好研究一下
        )

    return ((inputs, outputs),
            (loss, accuracy),
            (train_op, global_step, grad_input))


placeholders, metrics, others = create_model(
    hps, hps.vocab_size, 2
)

inputs, outputs = placeholders
loss, accuracy = metrics
train_op, global_step, grad_input = others

init_op = tf.global_variables_initializer()
train_keep_prob_value = 0.8
test_keep_prob_value = 1.0

num_train_steps = 10000
gradient_max_save = np.array(np.zeros((1,50)))
train_accuracy_save = []
test_accuracy_save = []
loss_save = []
# 验证集、测试集 输出函数
def eval_holdout(sess, dataset_for_test, batch_size):
    # 计算出 该数据集 有多少batch
    accuracy_vals = []
    loss_vals = []
    for i in range(int(Pa.TEST_DATA_NUM/hps.batch_size)):
        batch_labels = dataset_for_test[i*hps.batch_size:(i+1)*hps.batch_size,0]
        batch_inputs = dataset_for_test[i*hps.batch_size:(i+1)*hps.batch_size,1:]
        batch_labels = batch_labels.reshape(-1)
        accuracy_val, loss_val = sess.run([accuracy, loss],
                                          feed_dict={
                                              inputs: batch_inputs,
                                              outputs: batch_labels
                                          })
        accuracy_vals.append(accuracy_val)
        loss_vals.append(loss_val)
    return np.mean(accuracy_vals), np.mean(loss_vals)





# train: 99.7%
# valid: 92.7%
# test: 93.2%


with tf.Session() as sess:
    sess.run(init_op)
    for i in range(num_train_steps):
        batch_inputs,batch_labels = random_select_batch(hps.batch_size, new_train)
        batch_labels = batch_labels.reshape(-1)
        outputs_val = sess.run(
            [loss, accuracy, train_op, global_step],
            feed_dict={
                inputs: batch_inputs,
                outputs: batch_labels
            }
        )
        loss_val, accuracy_val, _, global_step_val = outputs_val
        #print('grad_input:',sess.run(grad_input,{inputs:batch_inputs,outputs:batch_labels}))
        gradient_mat = sess.run(grad_input, {inputs: batch_inputs, outputs: batch_labels})
        gradient_mat = np.array(gradient_mat)
        gradient_mat = gradient_mat.sum(axis=3)
        gradient_mat = gradient_mat.mean(axis=1)
        loss_save.append(loss_val)
        train_accuracy_save.append(accuracy_val)
        print(i + 1, '次训练!', 'loss:', loss_val, ' accuracy:', accuracy_val)
        if global_step_val % 100 == 0:
            gradient_max_save = np.concatenate((gradient_max_save,gradient_mat),axis=0)
            testdata_accuracy, testdata_loss = eval_holdout(sess, new_test, hps.batch_size)
            test_accuracy_save.append(testdata_accuracy)
            tf.logging.info(
                'Step: %5d, loss: %3.3f, accuracy: %3.3f'%(global_step_val, loss_val, accuracy_val))
            print('测试!', 'loss:', testdata_loss, ' accuracy:', testdata_accuracy)
    np.savez('LSTM_newoutputs_results.npz',gradient=gradient_max_save,
             train_accuracy=train_accuracy_save,
             test_accuracy_save=test_accuracy_save,
             loss=loss_save)
'''
INFO:tensorflow:Step: 10000, loss: 0.053, accuracy: 0.990
INFO:tensorflow: valid_data Step: 10000, loss: 0.661, accuracy: 0.88000
INFO:tensorflow: test_data Step: 10000, loss: 1.216, accuracy: 0.80000
'''
