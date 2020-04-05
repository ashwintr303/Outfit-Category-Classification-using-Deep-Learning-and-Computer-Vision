# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 23:33:18 2020

@author: AshwinTR
"""

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
from data-categorical import polyvore_dataset, DataGenerator
from utils import Config
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
import tensorflow.keras as tfk
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')

class MyBugFix(tf.keras.callbacks.Callback):
  def __init__(self, callbacks):
    self.callbacks = callbacks

  def on_epoch_end(self, epoch, logs=None):
    for callback in self.callbacks:
      callback()

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
              'learning_rate': Config['learning_rate'],
              'shuffle': True
              }

    train_generator =  DataGenerator(train_set, dataset_size, params)
    test_generator = DataGenerator(test_set, dataset_size, params)

    # Use GPU
    #strategy = tf.distribute.MirroredStrategy()
    #with strategy.scope():
    base_model = MobileNet(layers=tf.keras.layers, weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu',kernel_regularizer=regularizers.l2(Config['reg_value']), bias_regularizer=regularizers.l2(Config['reg_value']))(x)

    predictions = Dense(n_classes, activation = 'softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    #freeze all layers except last 2
    for layer in base_model.layers[:-2]:
        layer.trainable = False

    # define optimizers
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    plot_model(model, to_file='mobilenet_finetuned_model.png', show_shapes=True, show_layer_names=True)

    # training
    results = model.fit(train_generator,validation_data=test_generator,epochs=Config['num_epochs'], callbacks=[MyBugFix([train_generator.on_epoch_end])])
                        
    # save and plot the model 
    model.save('/home/ubuntu/hw4/mobilenet-model.hdf5')
    
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
    plt.title('Loss for finetuned MobileNet model')
    plt.legend()
    plt.savefig('/home/ubuntu/hw4/mobilenet_loss.png', dpi=256)
    plt.close()
    
    plt.plot(epochs, accuracy, label='acc')
    plt.plot(epochs, val_accuracy, label='val_acc')
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for finetuned MobileNetNet model')
    plt.legend()
    plt.savefig('/home/ubuntu/hw4/mobilenet_acc.png', dpi=256)
    plt.close()
    
    



