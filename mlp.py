import urllib.request
import os
import tarfile
import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras_preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding


def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)


def read_file(filetype):
    path = "C:/Users/36521/OneDrive/Desktop/MLexpr/aclImdb/"
    file_list = []

    positive_path = path + filetype + '/pos/'
    for f in os.listdir(positive_path):
        file_list += [positive_path + f]

    negative_path = path + filetype + '/neg/'
    for f in os.listdir(negative_path):
        file_list += [negative_path + f]

    print('read', filetype, 'files:', len(file_list))

    all_labels = ([1] * 12500 + [0] * 12500)
    all_texts = []

    for fi in file_list:
        with open(fi, encoding='utf8') as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]
    return all_labels, all_texts


y_train, train_text = read_file("train")
y_test, train_test = read_file("test")

y_train = np.array(y_train)
y_test = np.array(y_test)
test_text = train_test


token = Tokenizer(num_words=3800)
token.fit_on_texts(train_text)

x_train_seq = token.texts_to_sequences(train_text)
x_test_seq = token.texts_to_sequences(test_text)

x_train = sequence.pad_sequences(x_train_seq, maxlen=380)
x_test = sequence.pad_sequences(x_test_seq, maxlen=380)

# mlp
model = Sequential()
model.add(Embedding(output_dim=32, input_dim=3800, input_length=380))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(units=512, activation='relu'))  # 神经元节点数为512，激活函数为relu
model.add(Dropout(0.3))
model.add(Dense(units=256, activation='relu'))  # 神经元节点数为256，激活函数为relu
model.add(Dropout(0.35))
model.add(Dense(units=1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
train_history = model.fit(
    x=x_train, y=y_train, validation_split=0.2, epochs=15, batch_size=400, verbose=1)


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


show_train_history(train_history, 'accuracy', 'val_accuracy')
show_train_history(train_history, 'loss', 'val_loss')

scores = model.evaluate(x_test, y_test, batch_size=50, verbose=1)
# print(scores)
print('Test loss: ', scores[0])
print('Test accuracy: ', scores[1])
