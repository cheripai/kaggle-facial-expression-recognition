from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.datasets import cifar10
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam


class CNN:

    model = None

    def __init__(self, outputs, lr, decay, dropout):

        if type(self).__name__ == "VGG":
            base = VGG16(weights="imagenet", include_top=False)
        elif type(self).__name__ == "ResNet":
            base = ResNet50(weights="imagenet", include_top=False)
        elif type(self).__name__ == "Inception":
            base = InceptionV3(weights="imagenet", include_top=False)
        else:
            raise ValueError("Invalid model name: {}".format(type(self).__name__))

        self.model = self.add_top(base, outputs, dropout)
        opt = Adam(lr=lr, decay=decay)
        self.model.compile(optimizer=opt, loss="mse")


    def add_top(self, base_model, outputs, dropout):
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = Dense(512, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        predictions = Dense(outputs, activation="softmax")(x)
        model = Model(input=base_model.input, output=predictions)
        for layer in base_model.layers:
            layer.trainable = False
        return model


class VGG(CNN):
    def __init__(self, outputs, lr, decay, dropout=0.6):
        super(self.__class__, self).__init__(outputs, lr, decay, dropout)


class Inception(CNN):
    def __init__(self, outputs, lr, decay, dropout=0.5):
        super(self.__class__, self).__init__(outputs, lr, decay, dropout)


class ResNet(CNN):
    def __init__(self, outputs, lr, decay, dropout=0.5):
        super(self.__class__, self).__init__(outputs, lr, decay, dropout)
