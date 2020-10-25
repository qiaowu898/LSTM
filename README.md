# LSTM
I create a moudel contains LSTM and Bi_LSTM which are all coded with only tf1.8, the logic is totally realized by my own code
you will be more clear about this project by reading this file.

Qiyang_TEXT--->Layers.py:该文件，为各个脚本文件提供LSTM、RNN类以及各种变种进行调用。

LSTM_text_classification.py：调用Layers.py当中的LSTM_last类。

RNN_text_classification.py：调用Layers.py当中的RNN类。

Rnn_gradient_new_outputs.py：调用Layers.py当中的RNN类，RNN采用每个时间步的输出的综合来作为RNN处理模块的输出。

gradient_Bi_RNN.py：纯脚本设计，我没有抽象到Layers.py当中。

lstm_new_outputs.py：调用Layers.py当中的LSTM类。

multi_layer_BiLSTM.py：调用Layers.py当中的Bi_LSTM类多次，构造了多层双向LSTM的处理模块。
