# -*- coding:utf-8 -*-
import re

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from Parameters_Enum import parameters as Pa
from Qiyang_TEXT.Layers import LSTM_last

# 数据的读取
'''
使用tsv文件来存储数据，我们可以通过列标识字段来访问到相应的属性列。
'''
train = pd.read_csv(Pa.TRAIN_FILE, sep='\t')
test = pd.read_csv(Pa.TEST_FILE, sep='\t')
train_data = train.loc[:, 'data']
train_labels = train.loc[:, 'labels']
test_data = test.loc[:, 'data']
test_labels = test.loc[:, 'labels']


# lstm 需要的参数
def get_default_params():
    return tf.contrib.training.HParams(
        num_embedding_size=16,  # 词向量的维度
        num_timesteps=50,  # 时间步的长度
        max_len=50,  # 文本的单词数目，注意它要与num_timesteps的值保持相同
        vocab_size=6000,  # 单词->数值表示时的词典大小。单词种数超过6000将转化成0进行表示。
        num_lstm_nodes= 32,  # 第一层LSTM的模型表示维度与第二层LSTM模型表示维度
        num_fc_nodes=32,  # LSTM处理完之后，使用全连接层来进行后续处理
        batch_size=500,  # 一个batch所处理的样本数目
        clip_lstm_grads=1.0,  # 当梯度计算超过1时，令其为1
        learning_rate=0.001,  # 学习速率
        num_word_threshold=10,  # 过滤掉词频太少的单词，我们不会将这类单词转化为数值表示，而是直接化为0.10是在整个训练集当中统计的结果
    )


hps = get_default_params()  # 我们通过hps来调用相应的参数来进行使用


# 随机抽选数据的函数
def random_select_batch(batch_size, train):
    '''
    :param batch_size:一个epoch所选出的样本的数目
    :param train: 数据源
    :return: 随机挑选的数据结果
    '''
    row_rand_array = np.arange(train.shape[0])  # 对于数据源当中的n条数据，我们构造一个数组[0,1,2...n-1]
    np.random.shuffle(row_rand_array)  # 我们将row_rand_array数组当中的元素打乱
    batch_ = train[row_rand_array[0:batch_size]]  # 我们抽取断乱后的数组的前batch_size个元素所标识的数据来作为一个batch
    batch_labels = batch_[:, 0]  # 这个batch的标签列
    batch_labels = batch_labels.reshape(-1, 1)  # 为适应训练对于数据维度的要求，对数据的维度进行修改
    batch_data = batch_[:, 1:]  # 其它初标签列外的其它列作为数据参与到训练当中
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


train_data_prc = preprocess(train_data)  # 训练数据进行预处理
t = Tokenizer(num_words=hps.vocab_size)  # Tokenizer对象负责将单词转化为数值表示的结果，我们可以指定单词的数目
t.fit_on_texts(train_data_prc)  # t阅读我们预处理完之后文本数据，建立起单词到数值表示的映射关系
new_train_data = t.texts_to_sequences(train_data_prc)  # 将预处理后的文本转化为数值表示的结果，单词用数值[0,6000]进行表示。
new_train_sequence = keras.preprocessing.sequence.pad_sequences(
    new_train_data, maxlen=hps.max_len, padding='post', truncating='post'
)  # 我们的每一个文本，在转化为数值表示的结果后，每一条文本的数值数量也应当修剪为一样的长度。机器学习是不会接受数据的维度是变化不一的

# 对于测试集数据，我们进行同样的处理
test_data_prc = preprocess(test_data)
new_test_data = t.texts_to_sequences(test_data_prc)
new_test_sequence = keras.preprocessing.sequence.pad_sequences(new_test_data, maxlen=hps.max_len
                                                               , padding='post', truncating='post')

# 我们将处理完的数据的维度进行调整到适合训练所要求的维度
new_test_sequence = new_test_sequence.reshape(-1, hps.max_len)
np_train_labels = np.array(train_labels)
np_train_labels = np_train_labels.reshape(-1, 1)
new_test_labels = np.array(test_labels)
new_test_labels = new_test_labels.reshape(-1, 1)

# 分别将训练集和测试集的数据的标签与数据部分合并起来
new_train = np.concatenate((np_train_labels, new_train_sequence), axis=1)
new_test = np.concatenate((new_test_labels, new_test_sequence), axis=1)


lstm = LSTM_last(hps.num_embedding_size,hps.num_lstm_nodes,'Lstm_Moudel')

def create_model(hps, vocab_size, num_classes):
    '''
    构建lstm
    :param hps: 参数对象
    :param vocab_size:  词表 长度
    :param num_classes:  分类数目
    :return:
    '''
    num_timesteps = hps.num_timesteps  # 一个句子中 有 num_timesteps 个词语
    batch_size = hps.batch_size  # 一个epoch参与训练的数据样本的数目

    # 设置两个 placeholder， 内容id 和 标签id
    inputs = tf.placeholder(tf.int32, (batch_size, num_timesteps))  # 每个epoch抽象的输入数据
    outputs = tf.placeholder(tf.int32, (batch_size,))  # 每个epoch的抽象数据所对应的标签值

    global_step = tf.Variable(
        tf.zeros([], tf.int64),
        name='global_step',
        trainable=False)  # 用于保存当前训练的步数

    # 数据初始化，我们使用-1到1之间的均匀分布来产生数据
    embeding_initializer = tf.random_uniform_initializer(-1.0, 1.0)

    with tf.variable_scope('embedding', initializer=embeding_initializer):
        # 负责将数据当中的数值所表示的单词，扩展为用一个词向量来进行表示
        embeddings = tf.get_variable('embedding', [vocab_size, hps.num_embedding_size], tf.float32)

        # inputs的维度为[batch_size,numsteps],而经扩展后得到[batch_size,numsteps,dimension]
        embeding_inputs = tf.nn.embedding_lookup(embeddings, inputs)

    outputs = lstm.calculate(embeding_inputs)

    # 将outputs_p3传入到一个全连接层当中
    fc_init = tf.uniform_unit_scaling_initializer(factor=1.0)
    with tf.variable_scope('fc', initializer=fc_init):  # initializer 此范围内变量的默认初始值

        # 这种全连接层的使用方式非常简便，fc1就是我们得到的在该全连接层的每个单元的激活值
        fc1 = tf.layers.dense(outputs,
                              hps.num_fc_nodes,
                              activation=tf.nn.relu,
                              name='fc1')
        # 为避免过拟合，我们将将某些隐藏层单元给停用掉
        fc1_dropout = tf.nn.dropout(fc1, 0.8)  # 0.8表示我们保留可用的单元的比例

        # 对于输出层，我们只有一个单元，logits表示我们的输出结果[batch_size,1]，因为需要计算loss值，我们暂时不加sigmoid函数
        logits = tf.layers.dense(fc1_dropout,
                                 1,
                                 name='fc2')
        logits = tf.reshape(logits, (logits.shape[0],))
        outputs = tf.cast(outputs, tf.float32)

    with tf.name_scope('metrics'):
        binary_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits,
            labels=outputs
        )
        loss = tf.reduce_mean(binary_loss)

        # 为计算预测准确率，我们将0到1之间的激活值进行4舍5入，便可得到0、1标签
        y_pred = tf.round(
            tf.nn.sigmoid(logits)
        )

        # 完成准确率的计算
        correct_pred = tf.equal(tf.cast(outputs, tf.int32), tf.cast(y_pred, tf.int32))  # 这里也做了修改
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.name_scope('train_op'):
        tvars = tf.trainable_variables()  # 获取所有可以训练的变量
        for var in tvars:
            tf.logging.info('variable name: %s' % (var.name))  # 打印出所有可训练变量

        # 对求取的梯度值进行截断操作
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(loss, tvars),  # 计算loss对于各个变量进行反向求导的梯度值
            hps.clip_lstm_grads
        )

        # 计算当前的loss关于输入词的梯度，用它来反映不同位置的单词对于结果的影响力度
        grad_input = tf.gradients(loss, embeding_inputs)
        optimizer = tf.train.AdamOptimizer(hps.learning_rate)  # 对参数进行梯度修改
        train_op = optimizer.apply_gradients(
            zip(grads, tvars),  # 将 梯度和参数 绑定起来
            global_step=global_step
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

# 进行10000个epoch的训练
num_train_steps = 10000

# 用于保存训练过程当中的训练结果
gradient_max_save = np.array(np.zeros((1, 50)))
train_accuracy_save = []
test_accuracy_save = []
loss_save = []


#
def eval_holdout(sess, dataset_for_test, batch_size):
    '''
    该函数的主要功能是用测试集的数据进行测试
    :param sess: 用于调用开启计算流图的对象
    :param dataset_for_test: 测试数据集
    :param batch_size: 选取进行测试的数据的数目
    :return: 将准确率与loss值返回
    '''
    accuracy_vals = []  # 存储记录准确率值
    loss_vals = []  # 存储记录loss值
    for i in range(int(Pa.TEST_DATA_NUM / hps.batch_size)):
        # 我们将该batch的数据的标签和数据给抽取出来
        batch_labels = dataset_for_test[i * hps.batch_size:(i + 1) * hps.batch_size, 0]
        batch_inputs = dataset_for_test[i * hps.batch_size:(i + 1) * hps.batch_size, 1:]
        batch_labels = batch_labels.reshape(-1)

        # 获取准确率、loss值
        accuracy_val, loss_val = sess.run([accuracy, loss],
                                          feed_dict={
                                              inputs: batch_inputs,
                                              outputs: batch_labels
                                          })

        # 将当前batch的准确率和loss值进行存储记忆
        accuracy_vals.append(accuracy_val)
        loss_vals.append(loss_val)
        # 返回准确率均值、loss均值
    return np.mean(accuracy_vals), np.mean(loss_vals)


with tf.Session() as sess:
    sess.run(init_op)  # 首先对模型的参数进行初始化
    for i in range(num_train_steps):
        # 随机选择训练集数据当中的一个batch来进行训练
        batch_inputs, batch_labels = random_select_batch(hps.batch_size, new_train)
        batch_labels = batch_labels.reshape(-1)

        # 开启运行计算流图
        outputs_val = sess.run(
            [loss, accuracy, train_op, global_step],
            feed_dict={
                inputs: batch_inputs,
                outputs: batch_labels
            }
        )

        # 完成当前epoch的数据记录工作
        loss_val, accuracy_val, _, global_step_val = outputs_val
        gradient_mat = sess.run(grad_input, {inputs: batch_inputs, outputs: batch_labels})
        gradient_mat = np.array(gradient_mat)
        gradient_mat = gradient_mat.sum(axis=3)
        gradient_mat = gradient_mat.mean(axis=1)
        loss_save.append(loss_val)
        train_accuracy_save.append(accuracy_val)
        print(i + 1, '次训练!', 'loss:', loss_val, ' accuracy:', accuracy_val)

        # 每100次进行一次测试集数据的运算
        if global_step_val % 100 == 0:
            gradient_max_save = np.concatenate((gradient_max_save, gradient_mat), axis=0)
            testdata_accuracy, testdata_loss = eval_holdout(sess, new_test, hps.batch_size)
            test_accuracy_save.append(testdata_accuracy)
            print('测试!', 'loss:', testdata_loss, ' accuracy:', testdata_accuracy)

    np.savez('2layer_BiLSTM_results.npz', gradient=gradient_max_save,
             train_accuracy=train_accuracy_save,
             test_accuracy_save=test_accuracy_save,
             loss=loss_save)
