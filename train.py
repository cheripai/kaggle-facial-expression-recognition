from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from models.cnn import CNN

IMG_SIZE = 48

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.10,
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    "data/train", target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale", batch_size=32, class_mode="categorical", shuffle=True)

validation_generator = test_datagen.flow_from_directory(
    "data/valid", target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale", batch_size=1, class_mode="categorical", shuffle=False)

if __name__ == "__main__":
    cnn = CNN(3, (48, 48, 1), lr=0.001, dropout=0.2, decay=0.001)
    print(cnn.model.summary())
    callbacks = [
        TensorBoard(), ModelCheckpoint("models/weights.h5", monitor="val_loss", save_best_only=True, save_weights_only=True)
    ]
    cnn.model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=validation_generator.n,
        callbacks=callbacks)
