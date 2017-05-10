import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imsave

IMG_SIZE = 48
NUM_SAVE = 150000

datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.10, )

for i, (img, target) in enumerate(
        datagen.flow_from_directory(
            "train",
            target_size=(IMG_SIZE, IMG_SIZE),
            color_mode="grayscale",
            batch_size=1,
            class_mode="categorical",
            shuffle=True)):
    if i >= NUM_SAVE:
        break
    fname = str(i)+".png"
    target = np.argmax(target.squeeze())
    path = os.path.join("augmented", str(target))
    imsave(os.path.join(path, fname), img.squeeze())
