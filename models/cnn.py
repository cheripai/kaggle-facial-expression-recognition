from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.datasets import cifar10
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Input, Activation, merge, AveragePooling2D
from keras.models import Model
from keras.optimizers import Adam


class CNN:

    def __init__(self, outputs, input_shape, lr=0.001, decay=0, dropout=0):

        img_input = Input(shape=input_shape)
        x = Conv2D(64, (3, 3), padding="same")(img_input)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = Conv2D(128, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(128, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = Conv2D(256, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(256, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(256, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = Conv2D(512, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(512, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(512, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = Conv2D(512, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(512, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(512, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = Flatten()(x)
        x = Dense(1024)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(dropout)(x)
        x = Dense(1024)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(dropout)(x)
        predictions = Dense(outputs, activation="softmax")(x)

        self.model = Model(inputs=img_input, outputs=predictions)
        opt = Adam(lr=lr, decay=decay)
        self.model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])


class Inception_FCN:

    def __init__(self, outputs, input_shape, lr=0.001, decay=0, dropout=0):
        img_input = Input(shape=input_shape)
        x = self.inception_block(img_input)
        x = self.inception_block(x)
        x = self.inception_block(x)
        x = self.inception_block(x)
        x = Dropout(dropout)(x)
        x = Conv2D(outputs, 3, 3, border_mode="same")(x)
        x = GlobalAveragePooling2D()(x)
        predictions = Activation("softmax")(x)

        self.model = Model(inputs=img_input, outputs=predictions)
        opt = Adam(lr=lr, decay=decay)
        self.model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
        
    def Conv2D_bn(self, x, nb_filter, filter_size, subsample=(1, 1)):
        x = Conv2D(nb_filter, filter_size, filter_size, subsample=subsample, activation='relu', border_mode='same')(x)
        return BatchNormalization()(x)

    def inception_block(self, x):
        branch1x1 = self.Conv2D_bn(x, 32, 1, subsample=(2, 2))

        branch5x5 = self.Conv2D_bn(x, 24, 1)
        branch5x5 = self.Conv2D_bn(branch5x5, 32, 5, subsample=(2, 2))

        branch3x3dbl = self.Conv2D_bn(x, 32, 1)
        branch3x3dbl = self.Conv2D_bn(branch3x3dbl, 48, 3)
        branch3x3dbl = self.Conv2D_bn(branch3x3dbl, 48, 3, subsample=(2, 2))

        branch_pool = AveragePooling2D((3, 3), strides=(2, 2), border_mode="same")(x)
        branch_pool = self.Conv2D_bn(branch_pool, 16, 1)
        return merge([branch1x1, branch5x5, branch3x3dbl, branch_pool], mode="concat", concat_axis=-1)
