'''
乔家阳
上海交通大学
日期：2020年10月9日13:28:17
描述：该Layers模块是采用tensorflow1.8编写的，使用tensorflow1.x能够对代码进行更细致
     的控制。
'''
import  tensorflow as tf
import math
class LSTM():
    def __init__(self,word_dms,ln_nodes,name):
        '''
        :param word_dms:输入的词向量的维度
        :param ln_nodes: LSTM内部的向量的维度
        '''
        self.__word_dms = word_dms
        self.__ln_nodes = ln_nodes
        scale = 1.0 / math.sqrt(self.__word_dms + self.__ln_nodes) / 3.0
        lstm_init = tf.random_uniform_initializer(-scale, scale)
        with tf.variable_scope(name+'_lstm_parameters',initializer=lstm_init):
               #遗忘门的参数
               with tf.variable_scope('forget_gate'):
                       self.__fx,self.__fh,self.__fb = self.__generate_parameters()
               #输入门的参数
               with tf.variable_scope('input_gate'):
                       self.__ix,self.__ih,self.__ib = self.__generate_parameters()
               #输入控制的参数
               with tf.variable_scope('memory'):
                       self.__mx,self.__mh,self.__mb = self.__generate_parameters()
               #输出门的参数
               with tf.variable_scope('output_gate'):
                       self.__ox,self.__oh,self.__ob = self.__generate_parameters()
    def __generate_parameters(self):
        '''
        产生相应的对应于四个门的控制参数的参数
        :return:
        '''
        Wx = tf.get_variable('weights_x',[self.__word_dms,self.__ln_nodes])
        Wh = tf.get_variable('weights_h',[self.__ln_nodes,self.__ln_nodes])
        bias = tf.get_variable('bias',[1,self.__ln_nodes],initializer=tf.constant_initializer(0.0))
        return Wx,Wh,bias
    def calculate(self,input_data):
        '''
        :param input_data:LSTM模块要去处理的数据，它的维度为[batch_size,num_words,dms_word]
        :return:
        '''

        #LSTM模块中间所使用到的变量h,c，负责记忆中间计算的结果
        h = tf.Variable(tf.zeros([input_data.shape[0],self.__ln_nodes]),trainable=False)
        C = tf.Variable(tf.zeros([input_data.shape[0],self.__ln_nodes]),trainable=False)

        #outputs负责保存lstm运算的结果
        outputs = tf.Variable(tf.zeros(
            [input_data.shape[0],1,self.__ln_nodes]
        ),trainable=False)

        #对每个时间步进行计算
        for i in range(input_data.shape[1]):
            #将i时间步处的词向量提取出来，完成后续的计算
            vector = tf.reshape(input_data[:,i,:],[input_data.shape[0],input_data.shape[2]])
            forget_gate = tf.nn.sigmoid(
                tf.matmul(vector,self.__fx)+tf.matmul(h,self.__fh)+self.__fb
            )
            input_gate = tf.nn.sigmoid(
                tf.matmul(vector,self.__ix)+tf.matmul(h,self.__ih)+self.__ib
            )
            memory = tf.nn.tanh(
                tf.matmul(vector,self.__mx)+tf.matmul(h,self.__mh)+self.__mb
            )
            C = tf.multiply(C,forget_gate)+tf.multiply(input_gate,memory)
            output_gate = tf.nn.sigmoid(
                tf.matmul(vector,self.__ox)+tf.matmul(h,self.__oh)+self.__ob
            )
            h = tf.multiply(tf.nn.tanh(C),output_gate)

            #我们将h记录到outputs当中
            outputs = tf.concat([
                outputs,
                tf.reshape(h,[input_data.shape[0],1,self.__ln_nodes])
            ],axis=1)
        outputs = outputs[:,1:,:]
        return outputs
class Bi_LSTM():
    def __init__(self,word_dms,ln_nodes,name):
        '''
            :param word_dms:输入的词向量的维度
            :param ln_nodes: LSTM内部的向量的维度
            '''
        self.__word_dms = word_dms
        self.__ln_nodes = ln_nodes
        scale = 1.0 / math.sqrt(self.__word_dms + self.__ln_nodes) / 3.0
        lstm_init = tf.random_uniform_initializer(-scale, scale)
        with tf.variable_scope(name + '_lstm_posdirection', initializer=lstm_init):
            with tf.variable_scope('forget_gate'):
                self.__fx_pos, self.__fh_pos, self.__fb_pos = self.__generate_parameters()
            with tf.variable_scope('input_gate'):
                self.__ix_pos, self.__ih_pos, self.__ib_pos = self.__generate_parameters()
            with tf.variable_scope('memory'):
                self.__mx_pos, self.__mh_pos, self.__mb_pos = self.__generate_parameters()
            with tf.variable_scope('output_gate'):
                self.__ox_pos, self.__oh_pos, self.__ob_pos = self.__generate_parameters()
        with tf.variable_scope(name+'_lstm_negdirection',initializer=lstm_init):
            with tf.variable_scope('forget_gate'):
                self.__fx_neg, self.__fh_neg, self.__fb_neg = self.__generate_parameters()
            with tf.variable_scope('input_gate'):
                self.__ix_neg, self.__ih_neg, self.__ib_neg = self.__generate_parameters()
            with tf.variable_scope('memory'):
                self.__mx_neg, self.__mh_neg, self.__mb_neg = self.__generate_parameters()
            with tf.variable_scope('output_gate'):
                self.__ox_neg, self.__oh_neg, self.__ob_neg = self.__generate_parameters()
        with tf.variable_scope(name+'_lstm_pickout',initializer=tf.constant_initializer(1/(2*self.__ln_nodes))):
            self.pick_weights = tf.get_variable('weights',[2*self.__ln_nodes,self.__ln_nodes])
            self.pick_bias = tf.get_variable('bias',[1,self.__ln_nodes],initializer=tf.constant_initializer(0.0))
    def __generate_parameters(self):
        '''
        产生相应的对应于四个门的控制参数的参数
        :return:
        '''
        Wx = tf.get_variable('weights_x', [self.__word_dms, self.__ln_nodes])
        Wh = tf.get_variable('weights_h', [self.__ln_nodes, self.__ln_nodes])
        bias = tf.get_variable('bias', [1, self.__ln_nodes], initializer=tf.constant_initializer(0.0))
        return Wx, Wh, bias

    def calculate(self, input_data):
        '''
        :param input_data:LSTM模块要去处理的数据，它的维度为[batch_size,num_words,dms_word]
        :return:
        '''
        h_pos = tf.Variable(tf.zeros([input_data.shape[0], self.__ln_nodes]), trainable=False)
        C_pos = tf.Variable(tf.zeros([input_data.shape[0], self.__ln_nodes]), trainable=False)
        h_neg = tf.Variable(tf.zeros([input_data.shape[0], self.__ln_nodes]), trainable=False)
        C_neg = tf.Variable(tf.zeros([input_data.shape[0], self.__ln_nodes]), trainable=False)
        outputs_pos = tf.Variable(tf.zeros(
            [input_data.shape[0], 1, self.__ln_nodes]
        ), trainable=False)
        outputs_neg = tf.Variable(tf.zeros(
            [input_data.shape[0],1,self.__ln_nodes]
        ),trainable=False)
        for i in range(input_data.shape[1]):
            vector_pos = tf.reshape(input_data[:, i, :], [input_data.shape[0], input_data.shape[2]])
            vector_neg = tf.reshape(input_data[:,input_data.shape[1]-1-i,:],[input_data.shape[0],input_data.shape[2]])
            forget_gate_pos = tf.nn.sigmoid(
                tf.matmul(vector_pos, self.__fx_pos) + tf.matmul(h_pos, self.__fh_pos) + self.__fb_pos
            )
            forget_gate_neg = tf.nn.sigmoid(
                tf.matmul(vector_neg,self.__fx_neg) + tf.matmul(h_neg,self.__fh_neg) + self.__fb_neg
            )
            input_gate_pos = tf.nn.sigmoid(
                tf.matmul(vector_pos, self.__ix_pos) + tf.matmul(h_pos, self.__ih_pos) + self.__ib_pos
            )
            input_gate_neg = tf.nn.sigmoid(
                tf.matmul(vector_neg,self.__ix_neg) + tf.matmul(h_neg,self.__ih_neg) + self.__ib_neg
            )
            memory_pos = tf.nn.tanh(
                tf.matmul(vector_pos, self.__mx_pos) + tf.matmul(h_pos, self.__mh_pos) + self.__mb_pos
            )
            memory_neg = tf.nn.tanh(
                tf.matmul(vector_neg,self.__mx_neg) + tf.matmul(h_neg,self.__mh_neg) + self.__mb_neg
            )
            C_pos = tf.multiply(C_pos, forget_gate_pos) + tf.multiply(input_gate_pos, memory_pos)
            C_neg = tf.multiply(C_neg,forget_gate_neg) + tf.multiply(input_gate_neg,memory_neg)
            output_gate_pos = tf.nn.sigmoid(
                tf.matmul(vector_pos, self.__ox_pos) + tf.matmul(h_pos, self.__oh_pos) + self.__ob_pos
            )
            output_gate_neg = tf.nn.sigmoid(
                tf.matmul(vector_neg,self.__ox_neg) + tf.matmul(h_neg,self.__oh_neg) + self.__ob_neg
            )
            h_pos = tf.multiply(tf.nn.tanh(C_pos), output_gate_pos)
            h_neg = tf.multiply(tf.nn.tanh(C_neg),output_gate_neg)
            outputs_pos = tf.concat([
                outputs_pos,
                tf.reshape(h_pos, [input_data.shape[0], 1, self.__ln_nodes])
            ], axis=1)
            outputs_neg = tf.concat([
                tf.reshape(h_neg,[input_data.shape[0],1,self.__ln_nodes]),
                outputs_neg
            ],axis=1)
        outputs_pos = outputs_pos[:, 1:, :]
        outputs_neg = outputs_neg[:,:-1,:]
        outputs = tf.concat([outputs_pos,outputs_neg],axis=2)
        outputs_r = tf.reshape(outputs,[-1,outputs.shape[2]])
        outputs_ = tf.nn.tanh(
            tf.matmul(outputs_r,self.pick_weights)+self.pick_bias
        )
        outputs_res = tf.reshape(outputs_,[input_data.shape[0],input_data.shape[1],self.__ln_nodes])
        return outputs_res