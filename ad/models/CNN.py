import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers


class CNN(keras.Model):
    """Convolutional Neural Network Model"""

    def __init__(self, num_classes: int = 3, name=None, **kwargs):
        super().__init__(name=name)

        self.num_classes = num_classes

        # build the network model
        self.network = self.build_network(**kwargs.pop('network', {}))

    def call(self, x, **kwargs):
        return self.network(x, **kwargs)

    def build_network(self, input_shape: tuple, activation=tf.nn.relu, **kwargs):
        """Definition of the Network architecture"""

        input = Input(shape=input_shape, name="in")
        h = Conv2D(32, (3, 3), activation=activation, padding='same', name="conv1")(input)
        h = BatchNormalization(name='BN1')(h)  # Add BatchNormalization layer after convolutional layer
        h = MaxPooling2D((1, 2), name="maxpool1")(h)  # Update pooling kernel size to (1, 2)
        h = Conv2D(64, (3, 3), activation=activation, padding='same', name="conv2")(h)
        h = BatchNormalization(name='BN2')(h)  # Add BatchNormalization layer after convolutional layer
        h = MaxPooling2D((1, 2), name="maxpool2")(h)  # Update pooling kernel size to (1, 2)
        
        h = Flatten()(h)
        #h = Dense(1024, activation=activation, name="d0")(h)
        h = Dense(128, activation=activation, name="d1")(h)
        h = Dense(64, activation=activation, name="d3")(h)
        output = Dense(self.num_classes, activation='softmax', name="classifier_output")(h)

        return tf.keras.Model(inputs=input, outputs=output, name='Classifier')

    def train(self, X_train, y_train, X_val, y_val, batch_size=256, epochs=30):
        self.history = self.network.fit(X_train, y_train,
                                        validation_data=(X_val, y_val),
                                        batch_size=batch_size,
                                        epochs=epochs)
        self.history = self.history.history

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.network.evaluate(X_test, y_test)
        return loss, accuracy

    def predict(self, X):
        predictions = self.network.predict(X)
        return predictions

    def summary(self):
        self.network.summary()

