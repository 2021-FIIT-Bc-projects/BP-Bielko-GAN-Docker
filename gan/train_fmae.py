import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# imports
import sys
if 'google.colab' in sys.modules:
    from google.colab import drive
    drive.mount('/content/drive')
    output_path = "/content/drive/My Drive/gan_files"
    %rm -r /content/BP-Bielko-GAN-Docker
    %rm -r /content/src
    !git clone https://github.com/2021-FIIT-Bc-projects/BP-Bielko-GAN-Docker.git
    %cd BP-Bielko-GAN-Docker/gan
else:
    output_path = "."


from src.dcgan_models import *


# model definitions

model_name = "dcgan_128_test_64units"

p_dims = 100
p_n = 100
predetermined_inputs = np.random.randn(p_dims * p_n)  # n vectors from the normal distribution
predetermined_inputs = predetermined_inputs.reshape(p_n, p_dims)

height = 128
width = 128

generator = Generator(height, width, n_dim=100, n_paralell_samples=64, init_size=4)
discriminator = Discriminator(height, width, n_filters=128, dataset_path="dataset_download/thumbnails128x128")

gan = GAN(generator, discriminator, height=height, width=width, model_name=model_name, output_path=output_path)
