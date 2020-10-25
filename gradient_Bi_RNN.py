# -*- coding:utf-8 -*-

# 构建计算图-lstm
# embeding
# lstm
# fc
# train_op
# 训练流程代码
# 数据集的封装
#   api: next_batch(batch_size)： 输出的都是 id
# 词表封装
#   api：sentence2id(text_sentence):将句子转化为id
# 类别的封装：
#   api:category2id(text_category): 将类别转化为id

import tensorflow as tf
import os
import sys
import numpy as np
import math
import numpy as np
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
import pandas as pd
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn
import os
import sys
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import time
import tensorflow as tf
import keras
from sklearn.metrics import accuracy_score
import math
from Parameters_Enum import parameters as Pa
# 打印出 log
tf.logging.set_verbosity(tf.logging.INFO)

# 数据的读取
train = pd.read_csv(Pa.TRAIN_FILE, sep='\t')
test = pd.read_csv(Pa.TEST_FILE, sep='\t')
train_data = train.loc[:, 'data']
train_labels = train.loc[:, 'labels']
test_data = test.loc[:, 'data']
test_labels = test.loc[:, 'labels']
# lstm 需要的参数
def get_default_params():
    return tf.contrib.training.HParams(
        num_embedding_size = 16, # 每个词语的向量的长度

        # 指定 lstm 的 步长， 一个sentence中会有多少个词语
        # 因为执行的过程中是用的minibatch，每个batch之间还是需要对齐的
        # 在测试时，可以是一个变长的
        num_timesteps = 50, # 在一个sentence中 有 50 个词语
        max_len = 50,
        vocab_size = 6000,
        num_lstm_nodes = [32, 32], # 每一层的size是多少
        num_lstm_layers = 2, # 和上句的len 是一致的
        # 有 两层 神经单元，每一层都是 32 个 神经单元

        num_fc_nodes = 32, # 全连接的节点数
        batch_size = 500,
        clip_lstm_grads = 1.0,
        # 控制lstm的梯度，因为lstm很容易梯度爆炸或者消失
        # 这种方式就相当于给lstm设置一个上限，如果超过了这个上限，就设置为这个值
        learning_rate = 0.001,
        num_word_threshold = 10, # 词频太少的词，对于模型训练是没有帮助的，因此设置一个门限
    )
hps = get_default_params() # 生成 参数 对象

# 随机抽选数据的函数
def random_select_batch(batch_size, train):
    row_rand_array = np.arange(train.shape[0])
    np.random.shuffle(row_rand_array)
    batch_ = train[row_rand_array[0:batch_size]]
    batch_labels = batch_[:, 0]
    batch_labels = batch_labels.reshape(-1, 1)
    batch_data = batch_[:, 1:]
    return batch_data, batch_labels


# 数据的预处理
def clear_review(text):
    texts = []
    for item in text:
        item = item.replace("<br /><br />", "")
        item = re.sub("[^a-zA-Z]", " ", item.lower())
        texts.append(" ".join(item.split()))
    return texts


# 删除停用词+词形还原
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

train_data_prc = preprocess(train_data)
t = Tokenizer(num_words=hps.vocab_size)
t.fit_on_texts(train_data_prc)
new_train_data = t.texts_to_sequences(train_data_prc)  # 用序号表示单词
new_train_sequence = keras.preprocessing.sequence.pad_sequences(new_train_data, maxlen=hps.max_len
                                                                , padding='post', truncating='post')  # 将文本转换成统一长度n*200
test_data_prc = preprocess(test_data)
new_test_data = t.texts_to_sequences(test_data_prc)
new_test_sequence = keras.preprocessing.sequence.pad_sequences(new_test_data, maxlen=hps.max_len
                                                               , padding='post', truncating='post')  # 将文本转换成统一长度n*500
new_test_sequence = new_test_sequence.reshape(-1, hps.max_len)
np_train_labels = np.array(train_labels)
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
    num_timesteps = hps.num_timesteps # 一个句子中 有 num_timesteps 个词语
    batch_size = hps.batch_size

    # 设置两个 placeholder， 内容id 和 标签id
    inputs = tf.placeholder(tf.int32, (batch_size, num_timesteps))
    outputs = tf.placeholder(tf.int32, (batch_size, ))

    # dropout keep_prob 表示要keep多少值，丢掉的是1-keep_prob
    #keep_prob = tf.placeholder(tf.float32,name='keep_prob')

    global_step = tf.Variable(
        tf.zeros([], tf.int64),
        name='global_step',
        trainable = False)  # 可以保存 当前训练到了 哪一步，而且不训练

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
    scale = 1.0 / math.sqrt(hps.num_embedding_size + hps.num_lstm_nodes[-1]) / 3.0
    lstm_init = tf.random_uniform_initializer(-scale, scale)



    def _generate_parames_for_lstm_cell(x_size, h_size, bias_size):
        '''
        生成参数的变量
        :param x_size: x × w  其中 w 的形状
        :param h_size: 上一层 输出h 的形状
        :param bias_size: 偏置的形状
        :return: 各个 变量
        '''
        x_w = tf.get_variable('x_weights', x_size) # 输入x的w权重的值
        h_w = tf.get_variable('h_weights', h_size) # 上一层 输出h 的 值
        b = tf.get_variable('biases', bias_size, initializer=tf.constant_initializer(0.0)) # 偏置的 值

        return x_w, h_w, b




    with tf.variable_scope('rnn_nn', initializer = lstm_init):
        # 生成 四组 可变 参数，分别是 遗忘门、输入门、输出门  和 tanh
        # 输入门
        wx, wh, b = _generate_parames_for_lstm_cell( # 以i开头，代表 inputs
                x_size=[hps.num_embedding_size, hps.num_lstm_nodes[0]], # []
                h_size = [hps.num_lstm_nodes[0], hps.num_lstm_nodes[0]],
                bias_size = [1, hps.num_lstm_nodes[0]]
            )

        # 每个神经元的输出 形状同上
        with tf.variable_scope('rnn_back'):
            wx_back, wh_back, b_back = _generate_parames_for_lstm_cell(  # 以i开头，代表 inputs
                x_size=[hps.num_embedding_size, hps.num_lstm_nodes[0]],  # []
                h_size=[hps.num_lstm_nodes[0], hps.num_lstm_nodes[0]],
                bias_size=[1, hps.num_lstm_nodes[0]]
            )
        h_forward = tf.Variable(tf.zeros([batch_size, hps.num_lstm_nodes[0]]),
                        trainable = False
                        )
        h_backward = tf.Variable(tf.zeros([batch_size,hps.num_lstm_nodes[0]]),
                        trainable=False
                        )
        #用于记录每一个时间步输出结果的变量，不用于训练
        Rnn_outputs_forward = tf.Variable(tf.zeros([batch_size,1,hps.num_lstm_nodes[0]])
                                  ,trainable=False)
        Rnn_outputs_backward = tf.Variable(tf.zeros([batch_size,1,hps.num_lstm_nodes[0]]),
                                   trainable=False)
        for i in range(num_timesteps): # 按照 词语的数量 进行
            # [batch_size, 1, embed_size]
            embeding_input = embeding_inputs[:, i, :] # 取出 句子中 的 第一个词语(当i为1时)
            #这样每次取出来的 中间的那一维度 就是 1，可以将其合并掉

            # 因为是 只有一个词语，所以将其reshape成 二维
            embeding_input = tf.reshape(embeding_input, [batch_size, hps.num_embedding_size])

            # RNN内部的运算
            h_forward = tf.tanh(
                tf.matmul(embeding_input, wx) + tf.matmul(h_forward, wh) + b
            )
            Rnn_outputs_forward = tf.concat([Rnn_outputs_forward,tf.reshape(h_forward,[hps.batch_size,1,hps.num_lstm_nodes[0]])],axis=1)
        #last = h # 只需要 最后一个 输出 就可以了
        for k in range(num_timesteps):  # 按照 词语的数量 进行
                # [batch_size, 1, embed_size]
                i = num_timesteps-1-k
                embeding_input = embeding_inputs[:, i, :]  # 取出 句子中 的 第一个词语(当i为1时)
                # 这样每次取出来的 中间的那一维度 就是 1，可以将其合并掉

                # 因为是 只有一个词语，所以将其reshape成 二维
                embeding_input = tf.reshape(embeding_input, [batch_size, hps.num_embedding_size])

                # RNN内部的运算
                h_backward = tf.tanh(
                    tf.matmul(embeding_input, wx_back) + tf.matmul(h_backward, wh_back) + b_back
                )
                Rnn_outputs_backward = tf.concat(
                    [tf.reshape(h_backward, [hps.batch_size, 1, hps.num_lstm_nodes[0]]),Rnn_outputs_backward]
                    , axis=1)
                s=1

                #将第一个初始值为0的情况剔除掉
    Rnn_outputs_forward = Rnn_outputs_forward[:,1:,:]
    Rnn_outputs_backward = Rnn_outputs_backward[:,:-1,:]
    Rnn_outputs = tf.concat([Rnn_outputs_forward,Rnn_outputs_backward],axis=2)
    with tf.variable_scope('pick_weights',initializer=lstm_init):
             pick_weights = tf.get_variable('weights',[2*hps.num_lstm_nodes[0],hps.num_lstm_nodes[0]])
             pick_bias = tf.get_variable('bias',[1,hps.num_lstm_nodes[0]])
    with tf.variable_scope('output_weights',initializer=tf.constant_initializer(1/hps.num_timesteps)):
            output_weights = tf.get_variable('weights',[hps.num_timesteps,1])
            output_bias = tf.get_variable('bias',[1,hps.num_lstm_nodes[0]],initializer=tf.constant_initializer(0.0))

    Rnn_outputs_r = tf.reshape(Rnn_outputs,[-1,2*hps.num_lstm_nodes[0]])
    pick_outputs = tf.nn.tanh(tf.matmul(Rnn_outputs_r,pick_weights)+pick_bias)
    pick_outputs_r = tf.reshape(pick_outputs,[hps.batch_size,hps.num_timesteps,hps.num_lstm_nodes[0]])
    pick_outputs_rr = tf.reshape(pick_outputs_r,[hps.batch_size,hps.num_lstm_nodes[0],hps.num_timesteps])
    pick_outputs_rrr = tf.reshape(pick_outputs_rr,[-1,hps.num_timesteps])
    weighted_outputs = tf.matmul(pick_outputs_rrr,output_weights)
    weighted_outputs_1 = tf.reshape(weighted_outputs,[hps.batch_size,hps.num_lstm_nodes[0]]) + output_bias
    weighted_outputs_2 = tf.nn.tanh(weighted_outputs_1)

        # 输出 Tensor("lstm_nn/mul_149:0", shape=(100, 32), dtype=float32)
        # 和注释部分的 last 输出 是同样的结果


    # 将最后一层的输出 链接到一个全连接层上
    # 参考链接：https://www.w3cschool.cn/tensorflow_python/tensorflow_python-fy6t2o0o.html
    fc_init = tf.uniform_unit_scaling_initializer(factor=1.0)
    with tf.variable_scope('fc', initializer = fc_init): # initializer 此范围内变量的默认初始值
        fc1 = tf.layers.dense(weighted_outputs_2,
                              hps.num_fc_nodes,
                              activation = tf.nn.relu,
                              name = 'fc1')
        # 进行 dropout
        fc1_dropout = tf.nn.dropout(fc1, 0.8)
        # 进行更换 参考：https://blog.csdn.net/UESTC_V/article/details/79121642

        logits = tf.layers.dense(fc1_dropout, 1, name='fc2')
        logits = tf.reshape(logits,(logits.shape[0],))
        outputs = tf.cast(outputs,tf.float32)
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
        #print(type(outputs), type(y_pred))
        correct_pred = tf.equal(tf.cast(outputs,tf.int32), tf.cast(y_pred, tf.int32)) # 这里也做了修改
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.name_scope('train_op'):
        tvars = tf.trainable_variables() # 获取所有可以训练的变量
        for var in tvars:
            tf.logging.info('variable name: %s' % (var.name)) # 打印出所有可训练变量

        # 对 梯度进行 截断.
        # grads是截断之后的梯度
        grads, _ =         tf.clip_by_global_norm(
            tf.gradients(loss, tvars),  # 在可训练的变量的梯度
            hps.clip_lstm_grads
        )  # 可以 获得 截断后的梯度
        grad_input = tf.gradients(loss,embeding_inputs)
        optimizer = tf.train.AdamOptimizer(hps.learning_rate) # 将每个梯度应用到每个变量上去
        train_op = optimizer.apply_gradients(
            zip(grads, tvars), # 将 梯度和参数 绑定起来
            global_step = global_step # 这个参数 等会儿，再好好研究一下
        )


    return ((inputs, outputs),
            (loss, accuracy),
            (train_op, global_step,grad_input))

placeholders, metrics, others = create_model(
    hps, hps.vocab_size, 2
)

inputs, outputs = placeholders
loss, accuracy = metrics
train_op, global_step,grad_input = others

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
        if global_step_val % 100 == 0:
            gradient_max_save = np.concatenate((gradient_max_save,gradient_mat),axis=0)
            validdata_accuracy, validdata_loss = eval_holdout(sess, new_test, hps.batch_size)
            test_accuracy_save.append(validdata_accuracy)
            tf.logging.info(
                'Step: %5d, loss: %3.3f, accuracy: %3.3f'%(global_step_val, loss_val, accuracy_val))
        if global_step_val % 500 == 0:

            testdata_accuracy, testdata_loss = eval_holdout(sess, new_test, hps.batch_size)
            tf.logging.info(
                ' valid_data Step: %5d, loss: %3.3f, accuracy: %3.5f' % (global_step_val, validdata_loss, validdata_accuracy))
            tf.logging.info(
                ' test_data Step: %5d, loss: %3.3f, accuracy: %3.5f' % (global_step_val, testdata_loss, testdata_accuracy))
    np.save('gradient_Bi_rnn.npy',gradient_max_save)
    np.savez('Bi_RNN_results.npz',gradient=gradient_max_save,
             train_accuracy=train_accuracy_save,
             test_accuracy_save=test_accuracy_save,
             loss=loss_save)
'''
INFO:tensorflow:Step: 10000, loss: 0.053, accuracy: 0.990
INFO:tensorflow: valid_data Step: 10000, loss: 0.661, accuracy: 0.88000
INFO:tensorflow: test_data Step: 10000, loss: 1.216, accuracy: 0.80000
'''
