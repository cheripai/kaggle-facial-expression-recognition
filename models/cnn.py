from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.datasets import cifar10
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Input, Activation, concatenate, AveragePooling2D
from keras.models import Model
from keras.optimizers import Adam


class VGG:

    def __init__(self, outputs, input_shape, lr=0.001, decay=0, dropout=0):

        img_input = Input(shape=input_shape)
        x = self.Conv2D_bn(img_input, 64, 3)
        x = self.Conv2D_bn(x, 64, 3)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = self.Conv2D_bn(x, 128, 3)
        x = self.Conv2D_bn(x, 128, 3)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = self.Conv2D_bn(x, 256, 3)
        x = self.Conv2D_bn(x, 256, 3)
        x = self.Conv2D_bn(x, 256, 3)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = self.Conv2D_bn(x, 512, 3)
        x = self.Conv2D_bn(x, 512, 3)
        x = self.Conv2D_bn(x, 512, 3)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = Flatten()(x)
        x = self.Dense_bn(x, 1024)
        x = Dropout(dropout)(x)
        x = self.Dense_bn(x, 1024)
        x = Dropout(dropout)(x)
        predictions = Dense(outputs, activation="softmax")(x)

        self.model = Model(inputs=img_input, outputs=predictions)
        opt = Adam(lr=lr, decay=decay)
        self.model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    def Conv2D_bn(self, x, nb_filter, filter_size, strides=(1, 1)):
        x = Conv2D(nb_filter, (filter_size, filter_size), strides=strides, padding='same')(x)
        x = BatchNormalization()(x)
        return Activation("relu")(x)

    def Dense_bn(self, x, units):
        x = Dense(units)(x)
        x = BatchNormalization()(x)
        return Activation("relu")(x)


class Inception_FCN:
    def __init__(self, outputs, input_shape, lr=0.001, decay=0, dropout=0):
        img_input = Input(shape=input_shape)
        x = self.inception_block(img_input)
        x = self.inception_block(x)
        x = self.inception_block(x)
        x = self.inception_block(x)
        x = Dropout(dropout)(x)
        x = Conv2D(outputs, (3, 3), padding="same")(x)
        x = GlobalAveragePooling2D()(x)
        predictions = Activation("softmax")(x)

        self.model = Model(inputs=img_input, outputs=predictions)
        opt = Adam(lr=lr, decay=decay)
        self.model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    def Conv2D_bn(self, x, nb_filter, filter_size, strides=(1, 1)):
        x = Conv2D(nb_filter, (filter_size, filter_size), strides=strides, padding='same')(x)
        x = BatchNormalization()(x)
        return Activation("relu")(x)

    def inception_block(self, x):
        branch1x1 = self.Conv2D_bn(x, 64, 1, strides=(2, 2))

        branch5x5 = self.Conv2D_bn(x, 48, 1)
        branch5x5 = self.Conv2D_bn(branch5x5, 64, 5, strides=(2, 2))

        branch3x3dbl = self.Conv2D_bn(x, 64, 1)
        branch3x3dbl = self.Conv2D_bn(branch3x3dbl, 96, 3)
        branch3x3dbl = self.Conv2D_bn(branch3x3dbl, 96, 3, strides=(2, 2))

        branch_pool = AveragePooling2D((3, 3), strides=(2, 2), padding="same")(x)
        branch_pool = self.Conv2D_bn(branch_pool, 64, 1)
        return concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=-1)
