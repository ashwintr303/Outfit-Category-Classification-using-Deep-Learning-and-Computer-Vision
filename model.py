# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 20:33:18 2020

@author: AshwinTR
"""

import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from data-categorical import polyvore_dataset, DataGenerator
from utils import Config

#bug fix to shuffle data
class MyBugFix(tf.keras.callbacks.Callback):
  def __init__(self, callbacks):
    self.callbacks = callbacks

  def on_epoch_end(self, epoch, logs=None):
    for callback in self.callbacks:
      callback()

#Learning rate scheduling
class LearningRateScheduler(tf.keras.callbacks.Callback):
  def __init__(self, schedule):
    super(LearningRateScheduler, self).__init__()
    self.schedule = schedule

  def on_epoch_begin(self, epoch, logs=None):  
    lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
    scheduled_lr = self.schedule(epoch, lr)
    tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
    print('New learning rate')
    
  def lr_schedule(epoch,lr):
    if epoch > 11:
        lr = lr/2
    return lr  

#model architecture 
def my_model(width, height, depth, classes):
	model = Sequential()
	inputShape = (height, width, depth)
	chanDim = -1

	# first CONV => RELU => CONV => RELU => POOL layer set
	model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(Conv2D(32, (3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	# second CONV => RELU => CONV => RELU => POOL layer set
	model.add(Conv2D(64, (3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(Conv2D(64, (3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	# first (and only) set of FC => RELU layers
	model.add(Flatten())
	model.add(Dense(512,kernel_regularizer=regularizers.l2(Config['reg_value']),bias_regularizer=regularizers.l2(Config['reg_value'])))
	model.add(Activation("relu"))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))

	# softmax classifier
	model.add(Dense(classes,kernel_regularizer=regularizers.l2(Config['reg_value']), bias_regularizer=regularizers.l2(Config['reg_value'])))
	model.add(Activation("softmax"))

	# return the constructed network architecture
	return model


if __name__=='__main__':

    # data generators
    dataset = polyvore_dataset()
    transforms = dataset.get_data_transforms()
    X_train, X_test, y_train, y_test, n_classes = dataset.create_dataset()

    if Config['debug']:
        train_set = (X_train[:100], y_train[:100], transforms['train'])
        test_set = (X_test[:100], y_test[:100], transforms['test'])
        dataset_size = {'train': 100, 'test': 100}
    else:
        train_set = (X_train, y_train, transforms['train'])
        test_set = (X_test, y_test, transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}

    params = {'batch_size': Config['batch_size'],
              'n_classes': n_classes,
              'shuffle': True,
              'learning_rate': Config['learning_rate']
              }

    train_generator =  DataGenerator(train_set, dataset_size, params)
    test_generator = DataGenerator(test_set, dataset_size, params)

    # Use GPU
    model = my_model(width=224, height=224, depth=3, classes=n_classes)
    model.compile(optimizer=optimizers.RMSprop(lr=Config['learning_rate']), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    plot_model(model, to_file='category_model.png', show_shapes=True, show_layer_names=True)
    results = model.fit(train_generator,validation_data=test_generator,epochs=Config['num_epochs'],
                        callbacks=[MyBugFix([train_generator.on_epoch_end]),
                        LearningRateScheduler(lr_schedule)],shuffle=True)

    model.save('/home/ubuntu/hw4/category_model.hdf5')
    
    loss = results.history['loss']
    val_loss = results.history['val_loss']
    accuracy = results.history['accuracy']
    val_accuracy = results.history['val_accuracy']

    epochs = np.arange(len(loss))
    plt.figure()
    plt.plot(epochs, loss, label='loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.title('Loss for category model')
    plt.legend()
    plt.savefig('/home/ubuntu/hw4/category_model_loss.png', dpi=256)
    plt.close()
    
    plt.plot(epochs, accuracy, label='acc')
    plt.plot(epochs, val_accuracy, label='val_acc')
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for category model')
    plt.legend()
    plt.savefig('/home/ubuntu/hw4/category_model_acc.png', dpi=256)
    plt.close()



