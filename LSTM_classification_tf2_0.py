import pandas as pd
from keras.preprocessing.text import Tokenizer

%matplotlib inline
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import keras
from sklearn.metrics import accuracy_score
data = pd.read_csv('train_17000.tsv',sep='\t')
test = pd.read_csv('test_7500.tsv',sep='\t')
def clear_review(text):
    texts = []
    for item in text:
        item = item.replace("<br /><br />", "")
        item = re.sub("[^a-zA-Z]", " ", item.lower())
        texts.append(" ".join(item.split()))
    return texts
#删除停用词+词形还原
def stemed_words(text):
    stop_words = stopwords.words("english")
    lemma = WordNetLemmatizer()
    texts = []
    for item in text:
        words = [lemma.lemmatize(w, pos='v') for w in item.split() if w not in stop_words]
        texts.append(" ".join(words))
    return texts
#文本预处理
def preprocess(text):
    text = clear_review(text)
    text = stemed_words(text)
    return text
data_processed = preprocess(data.loc[:,'data'])
test_labels=test.loc[:,'labels']
test_processed = preprocess(test.loc[:,'data'])
print(type(data_processed))
data_processed = preprocess(data.loc[:,'data'])
test_labels=test.loc[:,'labels']
test_processed = preprocess(test.loc[:,'data'])
t = Tokenizer(num_words=6000)
t.fit_on_texts(data_processed+test_processed)
new_data = t.texts_to_sequences(data_processed)#用序号表示单词
new_sequence = keras.preprocessing.sequence.pad_sequences(new_data,maxlen=50
                ,padding='post',truncating='post')#将文本转换成统一长度n*500
model_lstm = keras.models.Sequential([
    keras.layers.Embedding(6000,32),
    keras.layers.LSTM(units=16,return_sequences=False),
    keras.layers.LSTM(units=16,return_sequences=False),
    keras.layers.Dense(16,activation='relu'),
    keras.layers.Dense(1,activation='sigmoid')
])
model_lstm.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history = model_lstm.fit(new_sequence,data.loc[:,'labels']
                    ,epochs=120)
new_test = t.texts_to_sequences(test_processed)#用序号表示单词
new_sequence_test = keras.preprocessing.sequence.pad_sequences(new_test,maxlen=200
                ,padding='post',truncating='post')#将文本转换成统一长度n*500
results = model_lstm.predict_classes(new_sequence_test)
print(accuracy_score(test_labels,results))