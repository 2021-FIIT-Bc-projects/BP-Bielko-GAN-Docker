# definicie funkcii

import numpy as np
import math
from matplotlib import pyplot as plt
from skimage.transform import resize
import random
from PIL import Image
import datetime # testing

model_type = "0_to_1_leakyReLU_tanh_x64"

default_width = 64
default_height = 64
pixel_depth = 3

dataset_size = 70000
dataset_path = "/content/ffhq-dataset/thumbnails128x128"
output_path = "/content/drive/My Drive/gan_files"

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


real_sample_dict = {}


def generate_real_samples(i_start, n):
    picked_sample_list = list()
    for i_image in range(i_start, i_start + n):
        chosen_sample = i_image

        if chosen_sample not in real_sample_dict:
            chosen_folder = chosen_sample - (chosen_sample % 1000)

            folder_string = str(chosen_folder)
            image_string = str(chosen_sample)
            folder_string = folder_string.rjust(5, '0')  # padding
            image_string = image_string.rjust(5, '0')  # padding

            full_path = dataset_path + '/' + folder_string + '/' + image_string + '.png'

            with Image.open(full_path) as image:
                image_array = np.array(image)
            image_array = resize(image_array, (default_height, default_width))

            real_sample_dict[chosen_sample] = image_array

        else:
            image_array = real_sample_dict[chosen_sample]
        picked_sample_list.append(image_array)

    # po nacitani n-vzoriek:
    X = np.array(picked_sample_list)  # .reshape(n, 1)
    y = np.ones((n, 1))
    return X, y
    # print(image_array)


def generate_real_samples_random(n, i_min, i_max):
    picked_sample_list = list()
    for i_image in range(n):
        chosen_sample = random.choice(range(i_min, i_max))
        # print(i_image, 'chose image', chosen_sample)

        if chosen_sample not in real_sample_dict:
            chosen_folder = chosen_sample - (chosen_sample % 1000)

            folder_string = str(chosen_folder)
            image_string = str(chosen_sample)
            folder_string = folder_string.rjust(5, '0')  # padding
            image_string = image_string.rjust(5, '0')  # padding

            full_path = dataset_path + '/' + folder_string + '/' + image_string + '.png'

            with Image.open(full_path) as image:
                image_array = np.array(image)
            image_array = resize(image_array, (default_height, default_width))

            real_sample_dict[chosen_sample] = image_array

        else:
            image_array = real_sample_dict[chosen_sample]
        picked_sample_list.append(image_array)

    # po nacitani n-vzoriek:
    X = np.array(picked_sample_list)  # .reshape(n, 1)
    y = np.ones((n, 1))
    return X, y
    # print(image_array)


def random_latent_points(n_dim, n):
    latent_vectors = np.random.randn(n_dim * n)  # n čísel z gauss. distrib.
    latent_vectors = latent_vectors.reshape(n, n_dim)
    return latent_vectors


def generate_fake_samples(generator, x_input, n_dim, n):  # [-1,1]
    X = generator.predict(x_input)
    y = np.zeros((n, 1))
    return X, y


def eval_performance(gan_model, generator, discriminator, losses, metadata_list, init_time,
                     n_dim, i_epoch, n_epochs, i_batch, n_batches, inputs, n=25, n_plot=10, plot_size=9):
    x_real, y_real = generate_real_samples_random(n, 0, dataset_size)
    _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)

    input_points = random_latent_points(n_dim, n)
    x_fake, y_fake = generate_fake_samples(generator, input_points, n_dim, n)
    _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)

    time_taken = datetime.datetime.now() - init_time

    eval_string = f'''[Epoch {i_epoch}/{n_epochs}, Batch {i_batch}/{n_batches}]
    Time since start: {time_taken}
    Disc. loss real: {losses[0]}
    Disc. loss fake: {losses[1]}
    Gen. loss: {losses[2]}
    Acc. real: {acc_real}
    Acc. fake: {acc_fake} (of {n} samples)
    '''

    with open(f'{output_path}/{model_type}/outputs/evaluation/epoch_metadata.txt', 'a+') as metadata_file:
        metadata_file.write(eval_string)
    print(eval_string)

    metadata_list.append(
        [
            time_taken.seconds / 3600,  # float hodnota kolko hodin od startu
            losses[0],
            losses[1],
            losses[2],
            acc_real,
            acc_fake
        ]
    )

    if i_batch % n_plot == 0:
        # n_factor = math.sqrt(n)
        fig = generate_and_plot(generator, n_dim, inputs, plot_size)
        epoch_padding_size = 8  # len(str(n_epochs-1))
        batch_padding_size = 8  # len(str(n_batches-1))
        filename = f"{output_path}/{model_type}/outputs/output_epoch_{str(i_epoch).rjust(epoch_padding_size, '0')}_{str(i_batch).rjust(batch_padding_size, '0')}.png"
        fig.savefig(filename)
        plt.show(fig)
        plt.close(fig)
        # input_points = random_latent_points(n_dim, n_factor * n_factor)
        # x_plt, _ = generate_fake_samples(generator, predetermined_inputs, n_dim, n_factor * n_factor)
        # padding_size = len(str(n_epochs-1))
        # fig = plt.figure(figsize=(n_factor*1.5, n_factor*1.5))
        # plt.text(0.5,-0.15,
        # f'''[Epoch {i_epoch}/{n_epochs}] Time since start: {datetime.datetime.now() - init_time}
        # Disc. loss real: {round(losses[0], 4)}, Disc. loss fake: {round(losses[1], 4)}, Gen. loss: {round(losses[2], 4)}
        # Acc. real: {round(acc_real, 4)}, Acc. fake: {round(acc_fake, 4)} (of {n} samples)
        # ''',
        #         horizontalalignment='center')
        #
        # fig.subplots_adjust(wspace=1/default_width,
        #                    hspace=1/default_height
        #                    )
        # for i in range(n_factor * n_factor):
        #    if i == 0:
        #        print(x_plt[i, default_height//2, default_width//2], '\n', ((x_plt[i, default_height//2, default_width//2]/2 + 0.5)*256).astype('int'))
        #
        #    # define subplot
        #    ax = fig.add_subplot(n_factor, n_factor, 1 + i)
        #    plt.axis('off')
        #    ax.imshow(float_to_rgb(x_plt)[i], interpolation='nearest')
        #
        # filename = f"{output_path}/{model_type}/outputs/generator_epoch_{str(i_epoch).rjust(padding_size,'0')}.png"
        # fig.savefig(filename)
        # plt.show()
        # plt.close(fig)


def train_gan(gan_model, generator, discriminator, dataset_size, metadata_list,
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
            x_real, y_real = generate_real_samples(start_n, half_batch)
            input_points = random_latent_points(n_dim, half_batch)
            x_fake, y_fake = generate_fake_samples(generator, input_points, n_dim, half_batch)

            d_loss_real, _ = discriminator.train_on_batch(x_real, y_real)
            d_loss_fake, _ = discriminator.train_on_batch(x_fake, y_fake)

            # vstup a target pre generator
            x_gan = random_latent_points(n_dim, n_batch)
            y_gan = np.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(x_gan, y_gan)

            if i % n_eval == 0:
                losses = (d_loss_real, d_loss_fake, g_loss)
                inputs = random_latent_points(n_dim, plot_size)
                eval_performance(gan_model, generator, discriminator, losses, metadata_list, init_time,
                                 n_dim, epoch, n_epochs, i, batches, inputs, n=eval_samples, n_plot=n_plot, plot_size=plot_size)

            start_n += half_batch;


def show_dataset():
    x_plt, _ = generate_real_samples_random(16, 0, dataset_size)
    n_factor = 3  # 4x4
    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(wspace=0, hspace=0)
    for i in range(n_factor * n_factor):  # ZMEN ABY NAMIESTO 4,5 BOLI FACTORS
        # define subplot
        ax = fig.add_subplot(n_factor, n_factor, 1 + i)
        plt.axis('off')
        ax.imshow(x_plt[i], interpolation='nearest')
    plt.show()
    plt.close(fig)


def generate_and_plot(generator, n_dim, inputs, n):
    n_factor = int(math.sqrt(n))
    x_plt, _ = generate_fake_samples(generator, inputs, n_dim, n)

    px = 1 / plt.rcParams['figure.dpi']
    fig = plt.figure(frameon=False, figsize=((n_factor * default_width) * px, (n_factor * default_height) * px))

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

    for i in range(n_dim):  # for each latent dim
        # generate a transition between A and B on this dim
        dim_transition = np.linspace(pointA[i], pointB[i], n_steps)
        for step in range(n_steps):  # then assign each step of transition properly to the final array
            transition_points[step][i] = dim_transition[step]

    return transition_points
