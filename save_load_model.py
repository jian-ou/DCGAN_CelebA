import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import DCGAN_discriminator
from DCGAN_discriminator import discriminator_DCGAN_model
import DCGAN_generator
from DCGAN_generator import generator_DCGAN_model

def save_model(
    model,
    save_dir = './data/mymodel.h5'):
    '''
    输入model名和保存的名字
    '''
    model.save(save_dir)
    print('\n成功保存模型参数到'+save_dir)

def load_model(save_dir = './data/mymodel.h5', show = 0):
    '''
    输入保存模型位置
    '''
    model = tf.keras.models.load_model(save_dir)
    if show != 0:
        model.summary()
    return model

def run():
    model = generator_DCGAN_model()
    model.summary()
    save_model(model)
    model = generator_DCGAN_model()
    model.summary()
    model = load_model(show=1)


if __name__ == "__main__":
    run()