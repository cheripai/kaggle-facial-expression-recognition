from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from models.cnn import Inception, ResNet, VGG


IMG_SIZE = 200


train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    # rotation_range=15,
    )
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    "data/train", target_size=(IMG_SIZE, IMG_SIZE), batch_size=8, class_mode="categorical", shuffle=True)

validation_generator = test_datagen.flow_from_directory(
    "data/valid", target_size=(IMG_SIZE, IMG_SIZE), batch_size=8, class_mode="categorical", shuffle=False)


if __name__ == "__main__":
    cnn = ResNet(3, lr=0.001, dropout=0.0, decay=0.001)
    callbacks = [TensorBoard()]
    cnn.model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=validation_generator.n,
        callbacks=callbacks)
    cnn.model.save_weights("models/weights.h5")
