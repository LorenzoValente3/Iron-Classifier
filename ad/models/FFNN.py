import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *

import tensorflow_addons as tfa
from typing import List


class FFNN(keras.Model):

    """FeedForward Neural Network Model"""

    def __init__(self, 
                 num_classes: int = 3, 
                 name=None, 
                 **kwargs):
        super().__init__(name=name)

        self.num_classes = num_classes
        self.history = None

        # build the network model
        self.network = self.build_network(**kwargs.pop('network', {}))

    def call(self, x, **kwargs):
        return self.network(x, **kwargs)

    def build_network(self, input_shape: tuple, layer_sizes: tuple = [128],
                   kernel_regularizer=None, #activation=tf.nn.relu,
                  BN: bool = False, PRelu: bool = False,
                  kernel_initializer=None, dropout_rate=None, **kwargs):
        """Definition of the Network architecture"""

        input = Input(shape=input_shape, name="input")
        layers = [input]

        # Create the layers
        for i, layer_size in enumerate(layer_sizes):
            layer = Dense(layer_size, #activation=activation, 
                        kernel_regularizer=kernel_regularizer, 
                        kernel_initializer=kernel_initializer,
                        name=f"d{i}")
            layers.append(layer(layers[-1]))

            if BN:
                # Add batch normalization layer after each dense layer
                bn = BatchNormalization(name=f"bn{i}")
                layers.append(bn(layers[-1]))      
                
            if PRelu:
                # Add PReLU activation layer after BatchNormalization
                prelu = PReLU(name=f"prelu{i}")
                layers.append(prelu(layers[-1]))
            
            if dropout_rate != None:
                # Add dropout layer after each dense layer
                dropout = Dropout(rate=dropout_rate, name=f"dropout{i}")
                layers.append(dropout(layers[-1]))

        output = Dense(self.num_classes, activation='softmax', name="classifier_output")(layers[-1])

        return tf.keras.Model(inputs=input, outputs=output, name='Classifier')



    def train(self, X_train, y_train, X_val, y_val, batch_size=256, epochs=30, callbacks=None):
        if callbacks is None:
            callbacks = []  # Initialize empty list if no callbacks are provided

        self.history = self.network.fit(X_train, y_train,
                                        validation_data=(X_val, y_val),
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        callbacks=callbacks)

        self.history = self.history.history

    def evaluate(self, X_test, y_test):
        loss, *accuracy = self.network.evaluate(X_test, y_test)
        return loss, accuracy

    def predict(self, X):
        predictions = self.network.predict(X)
        return predictions
    
    def summary(self):
        self.network.summary()



class EnsembleFFNN(keras.Model):
    def __init__(self, 
                 num_models=3, 
                 num_classes=3, 
                 name=None, 
                 **kwargs):
        super().__init__(name=name)

        self.num_models = num_models
        self.models = []
        self.num_classes = num_classes
        self.history = []

        for i in range(num_models):
            model = self.build_network(**kwargs.pop('network', {}))
            self.models.append(model)

    def build_network(self, input_shape: tuple = (1028,), layer_sizes=[128, 64], kernel_regularizer=None,
                      BN = True , PRelu = True, kernel_initializer=None, dropout_rate=None):
        input = Input(shape=input_shape, name="input")
        layers = [input]

        for i, layer_size in enumerate(layer_sizes):
            layer = Dense(layer_size, kernel_regularizer=kernel_regularizer,
                          kernel_initializer=kernel_initializer, name=f"d{i}")
            layers.append(layer(layers[-1]))

            if BN:
                bn = BatchNormalization(name=f"bn{i}")
                layers.append(bn(layers[-1]))

            if PRelu:
                prelu = PReLU(name=f"prelu{i}")
                layers.append(prelu(layers[-1]))

            if dropout_rate is not None:
                dropout = Dropout(rate=dropout_rate, name=f"dropout{i}")
                layers.append(dropout(layers[-1]))

        output = Dense(self.num_classes, activation='softmax', name="classifier_output")(layers[-1])

        return tf.keras.Model(inputs=input, outputs=output, name='Classifier')

    def train(self, X_train, y_train, X_val, y_val, batch_size=256, epochs=30, callbacks=None, optimizer='adam', loss='categorical_crossentropy'):
        if callbacks is None:
            callbacks = []

        self.history = []

        for i, model in enumerate(self.models):
            print(f"\n Training Model {i+1}")

            # Compile the model
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

            # Train the model
            history = model.fit(X_train, y_train,
                                validation_data=(X_val, y_val),
                                batch_size=batch_size,
                                epochs=epochs,
                                callbacks=callbacks)
            self.history.append(history.history)


    def evaluate(self, X_test, y_test):
        losses = []
        accuracies = []

        for model in self.models:
            loss, accuracy = model.evaluate(X_test, y_test)
            losses.append(loss)
            accuracies.append(accuracy)

        return losses, accuracies

    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        return np.mean(predictions, axis=0)
    
    def summary(self):
        for i, model in enumerate(self.models):
            print(f"Summary of Model {i+1}:")
            model.summary()
            print("\n")
