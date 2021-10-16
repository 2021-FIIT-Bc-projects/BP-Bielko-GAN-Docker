# imports
import math # pre 1d funkciu
import numpy as np
import tensorflow.keras.backend as k_backend
from tensorflow.keras.models import Sequential, load_model
# from keras.optimizers import *
from tensorflow.keras.layers import Dense, \
                         Conv2D, \
                         LeakyReLU, \
                         Dropout, \
                         Flatten, \
                         Reshape, \
                         Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.config import list_physical_devices

from tensorflow.keras import initializers, optimizers
from matplotlib import pyplot as plt
# from tensorflow_datasets.image import CelebA
from skimage.transform import resize
import random
from PIL import Image
import datetime # testing

import dcgan_models
import dcgan_functions