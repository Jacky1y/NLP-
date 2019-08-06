from __future__ import print_function
import numpy as np
np.random.seed(1122)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

# 设定参数
max_features = 10000   # 词汇表大小
# cut texts after this number of words (among top max_features most common words)
# 文本的长度（取最后部分，基于前 max_features 个常用词）
maxlen = 100
batch_size = 32   # 批数据量大小

# 载入数据
print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

# 裁剪为 maxlen 长度
print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

# 构建模型
print('Build model...')
model = Sequential()
# 嵌入层，每个词维度为128
model.add(Embedding(max_features, 128, dropout=0.1))
# LSTM层，输出维度128，可以尝试着换成 GRU 试试
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.1))
model.add(Dense(1))   # 单神经元全连接层
model.add(Activation('sigmoid'))   # sigmoid 激活函数层

model.summary()   # 模型概述

# try using different optimizers and different optimizer configs
# 这里可以尝试使用不同的损失函数和优化器
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 训练，迭代 15 次，使用测试集做验证（真正实验时最好不要这样做）
print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=15,
          validation_data=(X_test, y_test))

# 评估误差和准确率
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)