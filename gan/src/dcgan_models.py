# definicie modelu
import os.path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, \
                         Conv2D, \
                         LeakyReLU, \
                         Dropout, \
                         Flatten, \
                         Reshape, \
                         Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import initializers, optimizers

from os import stat, mkdir, path

import datetime # testing
import numpy as np
import math
from matplotlib import pyplot as plt
from skimage.transform import resize
import random
from PIL import Image

#dataset_path = "/content/ffhq-dataset/thumbnails128x128"
#output_path = "/content/drive/My Drive/gan_files"

class GAN:
    def __init__(self, generator, discriminator, width=64, height=64, model_name="dcgan_tanh_x64", output_path="", dataset_size=70000):
        self.dataset_size = dataset_size
    
        model_directory = path.join(output_path, model_name)
        try:
            stat(model_directory)
        except:
            mkdir(model_directory)
            mkdir(path.join(model_directory, "saves"))
            mkdir(path.join(model_directory, "outputs"))
            mkdir(path.join(model_directory, "anim"))

        self.height = height
        self.width = width
        self.model_name = model_name
        self.output_path = output_path
        
        discriminator.model.trainable = False
        self.model = Sequential()
    
        self.model.add(generator.model)
        self.model.add(discriminator.model)
    
        self.model.layers[0]._name = 'Generator'
        self.model.layers[1]._name = 'Discriminator'
    
        adam = Adam(lr=0.0002, beta_1=0.5)
        self.model.compile(loss='binary_crossentropy', optimizer=adam)
        
        self.generator = generator
        self.discriminator = discriminator
        
        

    
    def eval_performance(self, losses, metadata_list, init_time,
                     n_dim, i_epoch, n_epochs, i_batch, n_batches, inputs, n=25, n_plot=10, plot_size=9):
        
        x_real, y_real = self.discriminator.generate_real_samples_random(n, 0, self.dataset_size)
        _, acc_real = self.discriminator.model.evaluate(x_real, y_real, verbose=0)

        input_points = random_latent_points(n_dim, n)
        x_fake, y_fake = self.generator.generate_fake_samples(input_points, n_dim, n)
        _, acc_fake = self.discriminator.model.evaluate(x_fake, y_fake, verbose=0)

        time_taken = datetime.datetime.now() - init_time

        eval_string = f'''[Epoch {i_epoch}/{n_epochs}, Batch {i_batch}/{n_batches}]
        Time since start: {time_taken}
        Disc. loss real: {losses[0]}
        Disc. loss fake: {losses[1]}
        Gen. loss: {losses[2]}
        Acc. real: {acc_real}
        Acc. fake: {acc_fake} (of {n} samples)
        '''

        with open(path.join(self.output_path, self.model_name, 'outputs', 'evaluation', 'epoch_metadata.txt'), 'a+') as metadata_file:
            metadata_file.write(eval_string)
        print(eval_string)

        metadata_list.append(
            [
                i_epoch * n_batches + i_batch,  # poradove cislo batch
                losses[0],
                losses[1],
                losses[2],
                acc_real,
                acc_fake
            ]
        )

        if i_batch % n_plot == 0:
            # n_factor = math.sqrt(n)
            fig = generate_and_plot(self.generator, n_dim, inputs, plot_size)
            epoch_padding_size = 8  # len(str(n_epochs-1))
            batch_padding_size = 8  # len(str(n_batches-1))
            filename = path.join(self.output_path, self.model_name, "outputs", f"output_epoch_{str(i_epoch).rjust(epoch_padding_size, '0')}_" \
                       f"{str(i_batch).rjust(batch_padding_size, '0')}.png")
            fig.savefig(filename)
            plt.show(fig)
            plt.close(fig)


    def train_gan(self, dataset_size, metadata_list,
                  n_dim=100, start_epoch=0, n_epochs=100, n_batch=128, n_eval=2000, eval_samples=64, n_plot=10,
                  plot_size=9):
        # diskriminator updatujeme so vstupmi v pocte n_batch, pol. real, pol. fake
        half_batch = n_batch // 2
        batches = dataset_size // half_batch

        init_time = datetime.datetime.now()


        for epoch in range(start_epoch, n_epochs):
            start_n = 0  # pozicia v datasete pre epoch
            for i in range(batches):
                print(f"[Epoch {epoch}] Batch {i}/{batches}")

                # vstup a target pre diskriminator
                x_real, y_real = self.discriminator.generate_real_samples(start_n, half_batch)
                input_points = random_latent_points(n_dim, half_batch)
                x_fake, y_fake = self.generator.generate_fake_samples(input_points, n_dim, half_batch)

                d_loss_real, _ = self.discriminator.model.train_on_batch(x_real, y_real)
                d_loss_fake, _ = self.discriminator.model.train_on_batch(x_fake, y_fake)

                # vstup a target pre generator
                x_gan = random_latent_points(n_dim, n_batch)
                y_gan = np.ones((n_batch, 1))
                g_loss = self.model.train_on_batch(x_gan, y_gan)

                if i % n_eval == 0:
                    losses = (d_loss_real, d_loss_fake, g_loss)
                    inputs = random_latent_points(n_dim, plot_size)
                    self.eval_performance(losses, metadata_list, init_time,
                                     n_dim, epoch, n_epochs, i, batches, inputs, n=eval_samples, n_plot=n_plot, 
                                     plot_size=plot_size)

                start_n += half_batch
        
        
class Discriminator():
    def __init__(self, default_width, default_height, n_filters=128, pixel_depth=3, dataset_path=''):
        self.real_gen = ImageDataGenerator(horizontal_flip=True)
        self.dataset_path = dataset_path
        self.height = default_height;
        self.width = default_width;

        input_shape = (self.height, self.width, pixel_depth)

        self.model = Sequential()
    
        first_layer = Conv2D(  # vstupne np polia su sice 3d, ale convolution sa nad nimi robi 2d
            filters=n_filters,
            kernel_size=(3, 3),  # ^^
            strides=(2, 2),
            padding='same',
            input_shape=input_shape
        )
        first_activation = LeakyReLU(alpha=0.2)
        first_dropout = Dropout(0.1)
    
        second_layer = Conv2D(
            filters=n_filters * 2,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
        )
        second_activation = LeakyReLU(alpha=0.2)
        second_dropout = Dropout(0.1)
    
        third_layer = Conv2D(
            filters=n_filters * 4,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
        )
        third_activation = LeakyReLU(alpha=0.2)
        third_dropout = Dropout(0.1)
        
        fourth_layer = Conv2D(
            filters=n_filters * 4,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
        )
        fourth_activation = LeakyReLU(alpha=0.2)
        fourth_dropout = Dropout(0.1)
        
        flatten = Flatten()
        dropout = Dropout(0.4)
        output_dense = Dense(
            units=1,  # real/fake klasifikacia
            activation='sigmoid'
        )
    
        self.model.add(first_layer)
        self.model.add(first_activation)
        self.model.add(first_dropout)
        
        self.model.add(second_layer)
        self.model.add(second_activation)
        self.model.add(second_dropout)
        
        self.model.add(third_layer)
        self.model.add(third_activation)
        self.model.add(third_dropout)
        
        self.model.add(fourth_layer)
        self.model.add(fourth_activation)
        self.model.add(fourth_dropout)
    
        self.model.add(flatten)
        self.model.add(dropout)
        self.model.add(output_dense)
    
        # model.summary()
        adam = Adam(lr=0.0002, beta_1=0.5)
        self.model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])  # metrics kvoli evaluation

    #def generate_real_samples(self, n):
    #    # FINISH
    #
    #    picked_sample_list = list()
    #    for i_image in range(i_start, i_start + n):
    #        chosen_sample = i_image
    #        chosen_folder = chosen_sample - (chosen_sample % 1000)
    #
    #        folder_string = str(chosen_folder)
    #        image_string = str(chosen_sample)
    #        folder_string = folder_string.rjust(5, '0')  # padding
    #        image_string = image_string.rjust(5, '0')  # padding
    #
    #        full_path = dataset_path + '/' + folder_string + '/' + image_string + '.png'
    #
    #        with Image.open(full_path) as image:
    #            image_array = np.array(image)
    #        image_array = resize(image_array, (default_height, default_width))
    #
    #        real_sample_dict[chosen_sample] = image_array
    #
    #        picked_sample_list.append(image_array)
    #
    #    X = self.real_gen.flow_from_directory(self.dataset_path, class_mode=None, batch_size=n)
    #    y = np.ones((n, 1))
    #    return X,y
    
    def generate_real_samples(self, i_start, n):
        picked_sample_list = list()
        for i_image in range(i_start, i_start + n):
            chosen_sample = i_image
            chosen_folder = chosen_sample - (chosen_sample % 1000)
    
            folder_string = str(chosen_folder)
            image_string = str(chosen_sample)
            folder_string = folder_string.rjust(5, '0')  # padding
            image_string = image_string.rjust(5, '0')  # padding
    
            full_path = path.join(self.dataset_path, folder_string, image_string + '.png')
    
            with Image.open(full_path) as image:
                image_array = np.array(image)
            image_array = resize(image_array, (self.height, self.width))
            picked_sample_list.append(image_array)
    
        # after loading n samples:
        X = np.array(picked_sample_list)
        y = np.ones((n, 1))
        return X, y


    def generate_real_samples_random(self, n, i_min, i_max):
        picked_sample_list = list()
        for i_image in range(n):
            chosen_sample = random.choice(range(i_min, i_max))

            chosen_folder = chosen_sample - (chosen_sample % 1000)
    
            folder_string = str(chosen_folder)
            image_string = str(chosen_sample)
            folder_string = folder_string.rjust(5, '0')  # padding
            image_string = image_string.rjust(5, '0')  # padding
    
            full_path = path.join(self.dataset_path, folder_string, image_string + '.png')
    
            with Image.open(full_path) as image:
                image_array = np.array(image)
            image_array = resize(image_array, (self.height, self.width))
            picked_sample_list.append(image_array)
    
        # after loading n samples:
        X = np.array(picked_sample_list)  # .reshape(n, 1)
        y = np.ones((n, 1))
        return X, y


class Generator:
    
    def generate_fake_samples(self, x_input, n_dim, n):  # [-1,1]
        X = self.model.predict(x_input)
        y = np.zeros((n, 1))
        return X, y
    
    def __init__(self, default_height, default_width, n_dim=100, n_paralell_samples=256, pixel_depth=3):
        self.height = default_height;
        self.width = default_width;
        self.model = Sequential()

        first_layer = Dense(
            units=(default_height // 16) * (default_width // 16) * n_paralell_samples,
            input_dim=n_dim,
            # activation='linear'
        )
        first_activation = LeakyReLU(alpha=0.2)
        reshape = Reshape((default_height // 16, default_width // 16, n_paralell_samples))

        first_upsample = Conv2DTranspose(  # alternativne UpSample2D + Conv2D, zvacsenie a domyslenie, toto ich spaja do 1
            filters=n_paralell_samples // 2,
            kernel_size=(4, 4),  # idealne nasobok strides, inak moze nastat sachovnicovy vzor v convolution
            strides=(2, 2),
            # activation='linear',
            padding='same',
        )
        first_upsample_activation = LeakyReLU(alpha=0.2)

        second_upsample = Conv2DTranspose(  # alternativne UpSample2D + Conv2D, zvacsenie a domyslenie, toto ich spaja do 1
            filters=n_paralell_samples // 2,  # vygeneruje RGB values
            kernel_size=(4, 4),  # idealne nasobok strides, inak moze nastat sachovnicovy vzor v convolution
            strides=(2, 2),
            # activation='linear',
            padding='same',
        )
        second_upsample_activation = LeakyReLU(alpha=0.2)

        third_upsample = Conv2DTranspose(  # alternativne UpSample2D + Conv2D, zvacsenie a domyslenie, toto ich spaja do 1
            filters=n_paralell_samples // 2,  # vygeneruje RGB values
            kernel_size=(4, 4),  # idealne nasobok strides, inak moze nastat sachovnicovy vzor v convolution
            strides=(2, 2),
            # activation='linear',
            padding='same',
        )
        third_upsample_activation = LeakyReLU(alpha=0.2)

        fourth_upsample = Conv2DTranspose(  # alternativne UpSample2D + Conv2D, zvacsenie a domyslenie, toto ich spaja do 1
            filters=n_paralell_samples // 2,  # vygeneruje RGB values
            kernel_size=(4, 4),  # idealne nasobok strides, inak moze nastat sachovnicovy vzor v convolution
            strides=(2, 2),
            # activation='linear',
            padding='same',
        )
        fourth_upsample_activation = LeakyReLU(alpha=0.2)

        output_layer = Conv2D(
            filters=pixel_depth,  # rgb info
            kernel_size=(3, 3),
            activation='tanh',  # specialna akt. funk. pre rgb
            padding='same'
        )

        self.model.add(first_layer)
        self.model.add(first_activation)
        self.model.add(reshape)
        self.model.add(first_upsample)
        self.model.add(first_upsample_activation)
        self.model.add(second_upsample)
        self.model.add(second_upsample_activation)
        self.model.add(third_upsample)
        self.model.add(third_upsample_activation)
        self.model.add(fourth_upsample)
        self.model.add(fourth_upsample_activation)

        self.model.add(output_layer)

def rgb_to_float(rgb_value):
    zero_to_one = rgb_value / 256.0
    # normalized = (zero_to_one - 0.5) * 2
    return zero_to_one


def float_to_rgb(float_value):
    # converted_float = (float_value / 2) + 0.5
    rgb_value = (float_value * 256)
    rgb_value = np.where(rgb_value > 255, 255, rgb_value)
    rgb_value = np.where(rgb_value < 0, 0, rgb_value).astype('uint8')
    return rgb_value

def random_latent_points(n_dim, n):
    latent_vectors = np.random.randn(n_dim * n)  # n čísel z gauss. distrib.
    latent_vectors = latent_vectors.reshape(n, n_dim)
    return latent_vectors


def generate_and_plot(generator, n_dim, inputs, n):
    n_factor = int(math.sqrt(n))
    x_plt, _ = generator.generate_fake_samples(inputs, n_dim, n)

    px = 1 / plt.rcParams['figure.dpi']
    fig = plt.figure(frameon=False, figsize=((n_factor * generator.width) * px, (n_factor * generator.height) * px))

    for i in range(n_factor * n_factor):  # ZMEN ABY NAMIESTO 4,5 BOLI FACTORS
        # define subplot
        ax = fig.add_subplot(n_factor, n_factor, 1 + i)
        ax.axis('off')
        ax.imshow(float_to_rgb(x_plt)[i], interpolation='nearest')
    plt.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
    return fig


def latent_transition(pointA, pointB, n_dim=100, n_steps=100):
    transition_points = np.empty([n_steps, n_dim])


    for i in range(n_steps):
        step = (-math.cos(i / n_steps * math.pi) * 0.5 + 0.5)

        for dim in range(n_dim):
            transition_points[i][dim] = (pointB[dim] - pointA[dim]) * step + pointA[dim]

    return transition_points
