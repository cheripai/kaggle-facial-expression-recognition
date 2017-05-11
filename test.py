import numpy as np
import pandas as pd
import sys
from keras.preprocessing.image import ImageDataGenerator
from models.cnn import VGG, Inception_FCN

IMG_SIZE = 48
BATCH_SIZE = 256
OUTPUTS = 3
TEST_FILE = "data/test_data.csv"

test_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = test_datagen.flow_from_directory(
    "data/valid",
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=1,
    class_mode="categorical",
    shuffle=False)

def preprocess(img):
    img = img.reshape(IMG_SIZE, IMG_SIZE).astype("float32")
    img /= 255
    img = np.expand_dims(img, axis=0)
    return np.expand_dims(img, axis=-1)

def predict_ensemble(models, img):
    predictions = np.zeros((len(models), 3))
    for i in range(len(models)):
        predictions[i] = models[i].predict(img)
    return np.argmax(np.mean(predictions, axis=0)).squeeze()
    
if __name__ == "__main__":
    test_data = pd.read_csv(TEST_FILE, header=None).as_matrix()

    vgg = VGG(OUTPUTS, (IMG_SIZE, IMG_SIZE, 1), lr=0, dropout=0, decay=0)
    inception = Inception_FCN(OUTPUTS, (IMG_SIZE, IMG_SIZE, 1), lr=0, dropout=0, decay=0)
    vgg.model.load_weights(sys.argv[1])
    inception.model.load_weights(sys.argv[2])
    models = [vgg.model, inception.model]

    vgg_predictions = vgg.model.predict_generator(validation_generator, steps=validation_generator.n)
    inception_predictions = inception.model.predict_generator(validation_generator, steps=validation_generator.n)
    mean_predictions = np.argmax((vgg_predictions + inception_predictions) / 2, axis=1)
    targets = np.array([])
    for i, (img, target) in enumerate(validation_generator):
        if i >= validation_generator.n:
            break
        targets = np.concatenate((targets, np.argmax(target, axis=1)))
    print(np.sum(targets == mean_predictions) / validation_generator.n)

    with open("results/submission.txt", "w+") as f:
        f.write("Id,Category\n")
        for i in range(len(test_data)):
            img = preprocess(test_data[i, :])
            prediction = predict_ensemble(models, img)
            f.write("{},{}\n".format(i, prediction))
