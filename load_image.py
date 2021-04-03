import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import numpy as np

import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

def load_train_data(
    batch_size = 64, 
    img_height = 64, 
    img_width = 64, 
    data_dir = "../dataset/CelebA",
    validation=0.2,
    resize = "[-1, 1]"):
    data_dir = pathlib.Path(data_dir)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    if resize == "[-1, 1]":
        normalization_layer = layers.experimental.preprocessing.Rescaling(1./127.5, offset = -1)
    if resize == "[0, 1]":
        normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

    all_image_paths = list(data_dir.glob('*/*/*'))
    image_count = int(len(all_image_paths) * (1 - validation))
    epoch_size = int(image_count/batch_size)

    return train_ds, epoch_size

def run():
    train_ds, epoch_size = load_train_data()
    print(epoch_size)

if __name__ == "__main__":
    run()