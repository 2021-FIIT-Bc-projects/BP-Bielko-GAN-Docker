import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_name = input("Model name (e.g. gan_128): ")
goal = int(input("Goal epoch: "))


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
predetermined_inputs = np.load(f"{model_name}_inputs")

height = 128
width = 128

generator = Generator(height, width, n_dim=100, n_paralell_samples=64, init_size=4)
discriminator = Discriminator(height, width, n_filters=128, dataset_path="dataset_download/thumbnails128x128")

gan = GAN(generator, discriminator, height=height, width=width, model_name=model_name, output_path=output_path)

init_time = datetime.datetime.now()


generator.model.load_weights(f"{output_path}/{model_name}/saves/generator_{current}")
discriminator.model.load_weights(f"{output_path}/{model_name}/saves/discriminator_{current}")



for epoch in range(0, goal):
    print("Epoch", epoch)
    
    for step in range(0, 70000, 100):
        
        if step % 1000 == 0:
        filename = path.join(self.output_path, self.model_name, "outputs", f"output_epoch_{str(i_epoch).rjust(epoch_padding_size, '0')}_" \
                       f"{str(step % 1000).rjust(batch_padding_size, '0')}.png")
            fig.savefig(filename)
            if disable_plot == False:
        
            print(step)
            real_image, _ = self.encoder.generate_real_face_samples(step, 1, dataset_path="dataset_download/thumbnails128x128")
            decoded_real_image = self.model.predict(real_image)
            decoded_input_image = self.model.predict(input_image)
            fig = plt.imshow(decoded_real_image[0], interpolation='nearest')
            plt.show(fig)
            plt.close()
            fig = plt.imshow(decoded_input_image[0], interpolation='nearest')
            plt.show(fig)
            plt.close()
        
        samples, _ = self.encoder.generate_real_face_samples(step, 100, dataset_path="dataset_download/thumbnails128x128")
        self.model.fit(samples, samples, verbose=0)