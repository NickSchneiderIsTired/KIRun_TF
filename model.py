from tensorflow import keras
import tensorflow as tf


class Network(keras.Model):
    def __init__(self):
        super().__init__()
        self.layer1 = keras.layers.Dense(176, trainable=True)
        self.dropout1 = keras.layers.Dropout(0.9)
        #self.layer2 = keras.layers.Dense(44, trainable=True)
        self.classifier = keras.layers.Dense(2)

    @tf.function
    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.dropout1(x)
        #x = self.layer2(x)
        return self.classifier(x)
