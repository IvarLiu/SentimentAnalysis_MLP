import copy
import os
import re
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Embedding
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras_preprocessing import sequence
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA, KernelPCA


def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')  # 剔除html标签
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


token = Tokenizer(num_words=2000)
token.fit_on_texts(train_text)

x_train_seq = token.texts_to_sequences(train_text)
x_test_seq = token.texts_to_sequences(test_text)

x_train = sequence.pad_sequences(x_train_seq, maxlen=100)
x_test = sequence.pad_sequences(x_test_seq, maxlen=100)


model = Sequential()
model.add(Embedding(output_dim=32, input_dim=2000, input_length=100))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(units=1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
train_history = model.fit(
    x=x_train, y=y_train, validation_split=0.2, epochs=20, batch_size=300, verbose=1)


# 展示训练结果
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

scores = model.evaluate(x_test, y_test)
print('Test loss: ', scores[0])
print('Test accuracy: ', scores[1])

# visualization
pca_data = [i[:] for i in x_train]
pca = KernelPCA(n_components=3, kernel='cosine')
pca_data = pca.fit_transform(pca_data).tolist()
print('降维数据维度:', len(pca_data[0]))


y = y_test.copy()
y = np.array(y, dtype=bool)
pca_data = np.array(pca_data)

ax = plt.subplot(111, projection='3d')
ax.scatter(pca_data[y == 1, 0], pca_data[y == 1, 1],
           pca_data[y == 1, 2], c='red', label='Positive')
ax.scatter(pca_data[y == 0, 0], pca_data[y == 0, 1],
           pca_data[y == 0, 2], c='blue', label='Negative')
ax.legend(loc='best')
ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
plt.show()
