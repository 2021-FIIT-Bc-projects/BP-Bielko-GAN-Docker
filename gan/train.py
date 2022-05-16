# imports
output_path = "."
from src.dcgan_models import *


model_name = input("Model name (e.g. gan_128): ")
current = int(input("Current epoch: "))
goal = int(input("Goal epoch: "))
save_step = int(input("Save each <> epochs: "))



# model definitions

p_dims = 100
p_n = 100
predetermined_inputs = np.random.randn(p_dims * p_n)  # n vectors from the normal distribution
predetermined_inputs = predetermined_inputs.reshape(p_n, p_dims)

np.save(f"{model_name}_inputs", predetermined_inputs)





height = 128
width = 128

generator = Generator(height, width, n_dim=100, n_paralell_samples=64, init_size=4)
discriminator = Discriminator(height, width, n_filters=128, dataset_path="dataset_download/thumbnails128x128", lr=0.0002)

gan = GAN(generator, discriminator, height=height, width=width, model_name=model_name, output_path=output_path, inputs=predetermined_inputs, lr=0.0002)






print("Running...")

if current != 0:
    generator.model.load_weights(f"{output_path}/{model_name}/saves/generator_{current}")
    discriminator.model.load_weights(f"{output_path}/{model_name}/saves/discriminator_{current}")
    
    if input("Load saved input vectors? y/n: ") == 'y':
        predetermined_inputs = np.load(f"{model_name}_inputs")
    
    print(f"Loaded epoch {current}, continuing training...")

a = datetime.datetime.now()

while current < goal:
    
    from_epoch = current
    to_epoch = current+save_step
    current += save_step
    
    dataset_size = 70000
    
    
    
    gan.train_gan(dataset_size,
                    n_dim=100, start_epoch=from_epoch, n_epochs=to_epoch,
                    n_batch=40, n_eval=125, eval_samples=100, n_plot=500, plot_size=9, disable_plot=True)
    
    b = datetime.datetime.now()
    print("Time taken: ", b - a)
    
    generator.model.save_weights(f"{output_path}/{model_name}/saves/generator_{to_epoch}", overwrite=True)
    discriminator.model.save_weights(f"{output_path}/{model_name}/saves/discriminator_{to_epoch}", overwrite=True)


