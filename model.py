from tensorflow import keras
from keras import regularizers
import tensorflow as tf


class Network(keras.Model):
    def __init__(self):
        super().__init__()
        self.dropout1 = keras.layers.Dropout(0.5)
        self.layer1 = keras.layers.Dense(
            units=44,
            trainable=True,
            kernel_regularizer=regularizers.l1(0.0001),
        )
        self.dropout2 = keras.layers.Dropout(0.5)
        self.classifier = keras.layers.Dense(1, trainable=True)

    @tf.function
    def call(self, inputs):
        x = self.dropout1(inputs)
        x = self.layer1(x)
        x = self.dropout2(x)
        return self.classifier(x)
