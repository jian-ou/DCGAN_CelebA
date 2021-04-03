import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def discriminator_DCGAN_model():
    '''
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d (Conv2D)              (None, 32, 32, 64)        4864
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 32, 32, 64)        0
    _________________________________________________________________
    dropout (Dropout)            (None, 32, 32, 64)        0
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 16, 16, 128)       204928
    _________________________________________________________________
    leaky_re_lu_1 (LeakyReLU)    (None, 16, 16, 128)       0
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 16, 16, 128)       0
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 8, 8, 256)         819456
    _________________________________________________________________
    leaky_re_lu_2 (LeakyReLU)    (None, 8, 8, 256)         0
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 8, 8, 256)         0
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 4, 4, 512)         3277312
    _________________________________________________________________
    leaky_re_lu_3 (LeakyReLU)    (None, 4, 4, 512)         0
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 4, 4, 512)         0
    _________________________________________________________________
    flatten (Flatten)            (None, 8192)              0
    _________________________________________________________________
    dense (Dense)                (None, 1)                 8193      
    =================================================================
    Total params: 4,314,753
    Trainable params: 4,314,753
    Non-trainable params: 0
    _________________________________________________________________
    '''
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                input_shape=[64, 64, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

def run():
    model = discriminator_DCGAN_model()
    model.summary()

if __name__ == "__main__":
    run()