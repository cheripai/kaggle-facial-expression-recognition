from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from models.cnn import VGG, Inception_FCN, ResNet

IMG_SIZE = 48
BATCH_SIZE = 128
EPOCHS = 200
OUTPUTS = 3

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.10)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    "data/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True)

validation_generator = test_datagen.flow_from_directory(
    "data/valid",
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False)

if __name__ == "__main__":
    # vgg = VGG(OUTPUTS, (IMG_SIZE, IMG_SIZE, 1), lr=0.001, dropout=0.5, decay=0.0001)
    inception = Inception_FCN(OUTPUTS, (IMG_SIZE, IMG_SIZE, 1), lr=0.001, dropout=0.4, decay=0.001)
    # resnet = ResNet(OUTPUTS, (IMG_SIZE, IMG_SIZE, 1), lr=0.001, dropout=0.0, decay=0.001)
    cnns = [vgg, inception, resnet]
    for cnn in cnns:
        model_name = type(cnn).__name__
        print("Training {}".format(model_name))
        callbacks = [
            TensorBoard(), 
            ModelCheckpoint("results/{}.h5".format(model_name), monitor="val_acc", save_best_only=True, save_weights_only=True),
            EarlyStopping(monitor="val_acc", patience=20)
        ]
        cnn.model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.n // BATCH_SIZE + 1,
            epochs=EPOCHS,
            validation_data=validation_generator,
            validation_steps=validation_generator.n // BATCH_SIZE + 1,
            callbacks=callbacks)
