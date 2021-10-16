# definicie modelu

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, \
                         Conv2D, \
                         LeakyReLU, \
                         Dropout, \
                         Flatten, \
                         Reshape, \
                         Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers, optimizers

default_width = 64
default_height = 64
pixel_depth = 3

def make_discriminator(n_filters=128, input_shape=(default_width, default_height, pixel_depth)):
    discriminator = Sequential()

    first_layer = Conv2D(  # vstupne np polia su sice 3d, ale convolution sa nad nimi robi 2d
        filters=n_filters,
        kernel_size=(3, 3),  # ^^
        strides=(2, 2),
        padding='same',
        input_shape=input_shape
    )
    first_activation = LeakyReLU(alpha=0.2)

    second_layer = Conv2D(
        filters=n_filters * 2,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='same',
    )
    second_activation = LeakyReLU(alpha=0.2)

    third_layer = Conv2D(
        filters=n_filters * 4,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='same',
    )
    third_activation = LeakyReLU(alpha=0.2)

    flatten = Flatten()
    dropout = Dropout(0.4)
    output_dense = Dense(
        units=1,  # real/fake klasifikacia
        activation='sigmoid'
    )

    discriminator.add(first_layer)
    discriminator.add(first_activation)
    discriminator.add(second_layer)
    discriminator.add(second_activation)
    discriminator.add(third_layer)
    discriminator.add(third_activation)

    discriminator.add(flatten)
    discriminator.add(dropout)
    discriminator.add(output_dense)

    # discriminator.summary()
    adam = Adam(lr=0.0002, beta_1=0.5)
    discriminator.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])  # metrics kvoli evaluation
    return discriminator


def make_generator(n_dim=100, n_paralell_samples=256):
    generator = Sequential()

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

    generator.add(first_layer)
    generator.add(first_activation)
    generator.add(reshape)
    generator.add(first_upsample)
    generator.add(first_upsample_activation)
    generator.add(second_upsample)
    generator.add(second_upsample_activation)
    generator.add(third_upsample)
    generator.add(third_upsample_activation)
    generator.add(fourth_upsample)
    generator.add(fourth_upsample_activation)

    generator.add(output_layer)

    return generator


def make_gan_model(generator, discriminator):  # spoločný model cez ktorý sa môže generátor trénovať
    discriminator.trainable = False
    model = Sequential()

    model.add(generator)
    model.add(discriminator)

    model.layers[0]._name = 'Generator'
    model.layers[1]._name = 'Discriminator'

    adam = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=adam)
    return model



