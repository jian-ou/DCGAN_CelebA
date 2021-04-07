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
import progress
import save_load_model

#训练参数
EPOCHS = 300
BATCH_SIZE = 48
NOISE_DIM = 100
SAVE_TIME = 1
LOAD_MODEL = True

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM]) / 3.

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


#加载数据
train_dataset, epoch_size = load_train_data(batch_size=BATCH_SIZE, validation = 0.01)
#创建模型
generator = DCGAN_generator.generator_DCGAN_model()
discriminator = DCGAN_discriminator.discriminator_DCGAN_model()
#该方法返回计算交叉熵损失的辅助函数
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#定义Adam优化器
generator_optimizer = tf.keras.optimizers.Adam(2e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        batch_i = 0
        real = 0

        for image_batch,_ in dataset:
            train_step(image_batch)
            real = progress.print_progress_batch(epoch+1, batch_i, epoch_size, real)
            batch_i = batch_i + 1

    # 每 SAVE_TIME 个 epoch 保存一次模型
        if (epoch + 1) % SAVE_TIME == 0:
            save_load_model.save_model(generator, './data/generator.h5')
            save_load_model.save_model(discriminator, './data/discriminator.h5')

        print ('\rTime for epoch {} is {} sec                                                                               '.format(epoch + 1, time.time()-start))
        progress.save_time(time.time()-start)

if LOAD_MODEL == True:
    generator = save_load_model.load_model('./data/generator.h5')
    discriminator = save_load_model.load_model('./data/discriminator.h5')
    print("加载模型.......")
print("开始训练")
train(train_dataset, EPOCHS)