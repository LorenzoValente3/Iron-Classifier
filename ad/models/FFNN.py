import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from sklearn.model_selection import train_test_split

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
                  kernel_initializer=None, dropout_rate = None, **kwargs):
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
                prelu = PReLU(alpha_regularizer = kernel_regularizer,
                              name=f"prelu{i}")
                layers.append(prelu(layers[-1]))
            
            if dropout_rate != None:
                # Add dropout layer after each dense layer
                dropout = Dropout(rate=dropout_rate, name=f"dropout{i}")
                layers.append(dropout(layers[-1]))

        output = Dense(self.num_classes, activation='softmax', 
                       #kernel_regularizer=kernel_regularizer, 
                       kernel_initializer=kernel_initializer,
                       name="class_out")(layers[-1])

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

    """Ensemble FeedForward Neural Networks Models"""

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
                      BN = True , PRelu = True, kernel_initializer= 'he_uniform', dropout_rate=None):
        input = Input(shape=input_shape, name="input")
        layers = [input]

        for i, layer_size in enumerate(layer_sizes):
            layer = Dense(layer_size, 
                          kernel_regularizer=kernel_regularizer,
                          kernel_initializer=kernel_initializer, 
                          name=f"d{i}")
            layers.append(layer(layers[-1]))

            if BN:
                bn = BatchNormalization(name=f"bn{i}")
                layers.append(bn(layers[-1]))

            if PRelu:
                prelu = PReLU(alpha_regularizer = kernel_regularizer,
                              name=f"prelu{i}")
                layers.append(prelu(layers[-1]))

            if dropout_rate is not None:
                dropout = Dropout(rate=dropout_rate, name=f"dropout{i}")
                layers.append(dropout(layers[-1]))

        output = Dense(self.num_classes, 
                       kernel_initializer=kernel_initializer,
                       activation='softmax', name="classifier_output")(layers[-1])

        return tf.keras.Model(inputs=input, outputs=output, name='Classifier')

    def train(self, X_train, y_train, batch_size=256, epochs=30, callbacks=None, optimizer='adam', loss='categorical_crossentropy'):
        if callbacks is None:
            callbacks = []

        self.history = []
        metrics=['accuracy', 
                               keras.metrics.AUC(name='AUC'),
                               keras.metrics.Precision(name='prec'), 
                               keras.metrics.Recall(name='rec'),
                               tfa.metrics.F1Score(num_classes=3, average='weighted', name='f1'),
                                ]
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', verbose=True, patience=4,
                                              restore_best_weights=True)
        reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)
        model_checkpoint = keras.callbacks.ModelCheckpoint('./weights/MultiFFNN/model_weights.h5', 
                                        save_best_only=True,
                                        monitor='val_loss',
                                        mode='max',
                                        verbose=1)
        callbacks=[early_stop, model_checkpoint, reduceLR]
        
        for i, model in enumerate(self.models):
            print(f"\nTraining Model {i+1}")
            # Create a new train-validation split for each model
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.25,
                                                                                    random_state=i+1)

            # Compile the model
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

            # Train the model on the specific split
            history = model.fit(X_train_split, y_train_split,
                                validation_data=(X_val_split, y_val_split),
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
            print(loss, accuracies[:])

        return losses, accuracies

    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        return np.mean(predictions, axis=0)
    
    def summary(self):
        for i, model in enumerate(self.models):
            print(f"Summary of Model {i+1}:")
            model.summary()
            print("\n")
