import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def generator_DCGAN_model():
    ''' 
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense (Dense)                (None, 8192)              819200
    _________________________________________________________________
    batch_normalization (BatchNo (None, 8192)              32768
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 8192)              0
    _________________________________________________________________
    reshape (Reshape)            (None, 4, 4, 512)         0
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 8, 8, 256)         3276800
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 8, 8, 256)         1024
    _________________________________________________________________
    leaky_re_lu_1 (LeakyReLU)    (None, 8, 8, 256)         0
    _________________________________________________________________
    conv2d_transpose_1 (Conv2DTr (None, 16, 16, 128)       819200
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 16, 16, 128)       512
    _________________________________________________________________
    leaky_re_lu_2 (LeakyReLU)    (None, 16, 16, 128)       0
    _________________________________________________________________
    conv2d_transpose_2 (Conv2DTr (None, 32, 32, 64)        204800
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 32, 32, 64)        256
    _________________________________________________________________
    leaky_re_lu_3 (LeakyReLU)    (None, 32, 32, 64)        0
    _________________________________________________________________
    conv2d_transpose_3 (Conv2DTr (None, 64, 64, 3)         4800
    =================================================================
    Total params: 5,159,360
    Trainable params: 5,142,080
    Non-trainable params: 17,280
    _________________________________________________________________
    '''
    model = tf.keras.Sequential()
    model.add(layers.Dense(4*4*512, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((4, 4, 512)))
    assert model.output_shape == (None, 4, 4, 512) # 注意：batch size 没有限制

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 64, 64, 3)

    return model

def run():
    model = generator_DCGAN_model()
    model.summary()

if __name__ == "__main__":
    run()