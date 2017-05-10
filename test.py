import sys
from keras.preprocessing.image import ImageDataGenerator
from models.cnn import VGG, Inception_FCN

IMG_SIZE = 48
BATCH_SIZE = 256

test_datagen = ImageDataGenerator(rescale=1. / 255)


validation_generator = test_datagen.flow_from_directory(
    "data/valid",
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False)
test_generator = test_datagen.flow_from_directory(
    "data/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False)

if __name__ == "__main__":
    # cnn = VGG(3, (48, 48, 1), lr=0, dropout=0, decay=0)
    cnn = Inception_FCN(3, (48, 48, 1), lr=0, dropout=0, decay=0)
    cnn.model.load_weights(sys.argv[1])
    print(cnn.model.evaluate_generator(validation_generator, steps=validation_generator.n // BATCH_SIZE + 1))
