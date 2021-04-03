import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
from tensorflow.keras import layers
import time
import cv2 as cv
import sys
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from IPython import display

#加载自己的文件
import load_image
from load_image import load_train_data
import DCGAN_discriminator
import DCGAN_generator
import print_progress
import save_load_model

SHOW_SIZE_BIG = False
epoch = 70

def run():
    model = save_load_model.load_model('./data/generator.h5')
    noise = tf.random.normal([9, 100]) / 3.
    #print(np.max(noise), np.min(noise))
    output_images = model(noise)
    if SHOW_SIZE_BIG == True:
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            image = (output_images[i] + 1) / 2
            #print(image)
            #print(np.max(image), np.min(image))
            plt.imshow(image)
            plt.axis("off")
    else:
        image = (output_images[5] + 1) / 2
        plt.imshow(image)
        plt.axis("off")
        plt.savefig('./data/image/test{:04d}.png'. format(epoch))
    plt.show()

if __name__ == "__main__":
    run()