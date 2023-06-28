import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from sklearn.model_selection import train_test_split

import tensorflow_addons as tfa

class EnsembleFFNN(keras.Model):
    """Ensemble FeedForward Neural Networks Models"""

    def __init__(self, 
                        num_models: int = 1, 
                        num_classes: int = 3,  
                        learning_rates: float = None, 
                        batch_sizes: int = None, 
                        kernel_initializers = None,
                        name = None,
                        **kwargs):
        super().__init__(name=name)

        self.num_models = num_models
        self.models = []
        self.num_classes = num_classes
        self.history = []

        if learning_rates is None:
            learning_rates = [0.001] * num_models  # Default learning rate is 0.001 for all models

        if batch_sizes is None:
            batch_sizes = [256] * num_models  # Default batch size is 256 for all models

        if kernel_initializers is None:
            kernel_initializers = ['he_uniform'] * num_models  # Default kernel initializer is 'he_uniform' for all models

        self.learning_rates = learning_rates
        self.batch_sizes = batch_sizes
        self.kernel_initializers = kernel_initializers

        for i in range(num_models):
            model = self.build_network(**kwargs.pop('network', {}),
                                      kernel_initializer=kernel_initializers[i]
                                       )
            self.models.append(model)

        self.evaluation_metrics = ['accuracy', 
                        keras.metrics.AUC(name='AUC'),
                        keras.metrics.Precision(name='prec'), 
                        keras.metrics.Recall(name='rec'),
                        tfa.metrics.F1Score(num_classes=self.num_classes, average='weighted', name='f1')]

    def build_network(self, input_shape=(1028,), layer_sizes=[128, 64], kernel_regularizer=None,
                      BN=True, PRelu=True, kernel_initializer='he_uniform', dropout_rate=None):
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
                prelu = PReLU(alpha_regularizer=kernel_regularizer,
                              name=f"prelu{i}")
                layers.append(prelu(layers[-1]))

            if dropout_rate is not None:
                dropout = Dropout(rate=dropout_rate, name=f"dropout{i}")
                layers.append(dropout(layers[-1]))

        output = Dense(self.num_classes,
                       kernel_initializer=kernel_initializer,
                       activation='softmax', name="classifier_output")(layers[-1])

        return tf.keras.Model(inputs=input, outputs=output, name='Classifier')

    def train(self, X, y, epochs=30, callbacks=None, loss='categorical_crossentropy'):
        
        #early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', verbose=True, patience=4,
        #                                      restore_best_weights=True)
        #reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)
        #model_checkpoint = keras.callbacks.ModelCheckpoint('./weights/MultiFFNN/model_weights.h5', 
        #                                save_best_only=True,
        #                                monitor='val_loss',
        #                                mode='max',
        #                                verbose=1)
        #callbacks=[early_stop, model_checkpoint, reduceLR]

        self.history = []

        for i, model in enumerate(self.models):
            print(f"\nTraining Model {i+1}")
            # Create a new train-validation split for each model
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X, y, 
                                                                                      test_size=0.25,
                                                                                      random_state=i+1)

            # Compile the model
            model.compile(optimizer='adam',#tf.keras.optimizers.Adam(self.learning_rates[i]), 
                          loss=loss, 
                          metrics=self.evaluation_metrics)

            # Train the model on the specific split
            history = model.fit(X_train_split, y_train_split,
                                validation_data=(X_val_split, y_val_split),
                                batch_size=self.batch_sizes[i],
                                epochs=epochs,
                                callbacks=callbacks)
            self.history.append(history.history)

            # Save the model
            self.save_model(model, i)

    def save_model(self, model, index):
        folder_path = f"./weights/MultiFFNN/model_{index+1}_latest"
        os.makedirs(folder_path, exist_ok=True)

        # Save model architecture
        model_architecture_path = os.path.join(folder_path, 'model_architecture.json')
        with open(model_architecture_path, 'w') as f:
            f.write(model.to_json())

        # Save model weights
        model_weights_path = os.path.join(folder_path, 'model_weights.h5')
        model.save_weights(model_weights_path)

    def evaluate(self, X_test, y_test):
        losses = []
        accuracies = []

        for model in self.models:
            result = model.evaluate(X_test, y_test)
            if isinstance(result, float):
                loss = result
                accuracy = None
            else:
                loss, accuracy = result

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
