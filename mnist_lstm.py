'''This is a reproduction of the IRNN experiment
with pixel-by-pixel sequential MNIST in
"A Simple Way to Initialize Recurrent Networks of Rectified Linear Units"
by Quoc V. Le, Navdeep Jaitly, Geoffrey E. Hinton

arxiv:1504.00941v2 [cs.NE] 7 Apr 2015
http://arxiv.org/pdf/1504.00941v2.pdf

Optimizer is replaced with RMSprop which yields more stable and steady
improvement.

Reaches 0.93 train/test accuracy after 900 epochs
(which roughly corresponds to 1687500 steps in the original paper.)
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import SimpleRNN, LSTM
from keras import initializers
from keras.optimizers import RMSprop
import os

model_name = 'LSTM'
if not os.path.exists(os.path.join('expr', model_name)):
    os.system('mkdir {}'.format(os.path.join('expr', model_name)))
batch_size = 32
num_classes = 10
epochs = 200
hidden_units = 100

learning_rate = 1e-4
clip_norm = 1.0

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1, 1)
x_test = x_test.reshape(x_test.shape[0], -1, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print('Evaluate Model {}...'.format(model_name))
model = Sequential()
model.add(LSTM(units=hidden_units, 
    activation='tanh', 
    kernel_initializer='glorot_uniform', 
    recurrent_initializer='orthogonal', 
    input_shape=x_train.shape[1:]))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

train_log = keras.callbacks.CSVLogger(os.path.join('expr', model_name, 'training.log'))

ckpt_saver = keras.callbacks.ModelCheckpoint(os.path.join('expr', model_name, 'weights.{epoch:02d}-{acc: .4f}.hdf5'), 
        monitor='acc', 
        verbose=1, 
        period=10)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[train_log, ckpt_saver])

scores = model.evaluate(x_test, y_test, verbose=0)
print('Model {} test score: {}'.format(model_name, scores[0]))
print('Model {} test accuracy: {}'.format(model_name, scores[1]))
