from keras.applications.resnet50 import identity_block, conv_block
from keras.layers import Activation, AveragePooling2D, BatchNormalization, concatenate, Conv2D
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, Input, MaxPooling2D, ZeroPadding2D
from keras.models import Model
from keras.optimizers import Adam


class CNN:

    model = None

    def __init__(self, lr=0.001, decay=0):
        opt = Adam(lr=lr, decay=decay)
        self.model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    def Conv2D_bn(self, x, nb_filter, filter_size, strides=(1, 1), padding="same"):
        x = Conv2D(nb_filter, (filter_size, filter_size), strides=strides, padding=padding)(x)
        x = BatchNormalization()(x)
        return Activation("relu")(x)


class VGG(CNN):
    def __init__(self, outputs, input_shape, lr=0.001, decay=0, dropout=0):
        img_input = Input(shape=input_shape)
        x = self.vgg_block(img_input, 2, 64)
        x = self.vgg_block(x, 2, 128)
        x = self.vgg_block(x, 3, 256)
        x = self.vgg_block(x, 3, 512)
        x = Flatten()(x)
        x = self.Dense_bn(x, 1024)
        x = Dropout(dropout)(x)
        x = self.Dense_bn(x, 1024)
        x = Dropout(dropout)(x)
        predictions = Dense(outputs, activation="softmax")(x)
        self.model = Model(inputs=img_input, outputs=predictions)
        super().__init__(lr, decay)

    def vgg_block(self, x, nb_conv, nb_filters, filter_size=3):
        for i in range(nb_conv):
            x = self.Conv2D_bn(x, nb_filters, filter_size)
        return MaxPooling2D((2, 2), strides=(2, 2))(x)
            
    def Dense_bn(self, x, units):
        x = Dense(units)(x)
        x = BatchNormalization()(x)
        return Activation("relu")(x)


class Inception_FCN(CNN):
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
        super().__init__(lr, decay)

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


class ResNet(CNN):
    def __init__(self, outputs, input_shape, lr=0.001, decay=0, dropout=0):
        img_input = Input(shape=input_shape)

        x = conv_block(img_input, 3, [32, 32, 128], stage=1, block="a", strides=(1, 1))
        x = identity_block(x, 3, [32, 32, 128], stage=1, block="b")
        x = identity_block(x, 3, [32, 32, 128], stage=1, block="c")

        x = conv_block(x, 3, [64, 64, 256], stage=2, block="a")
        x = identity_block(x, 3, [64, 64, 256], stage=2, block="b")
        x = identity_block(x, 3, [64, 64, 256], stage=2, block="c")

        x = conv_block(x, 3, [128, 128, 512], stage=3, block="a")
        x = identity_block(x, 3, [128, 128, 512], stage=3, block="b")
        x = identity_block(x, 3, [128, 128, 512], stage=3, block="c")

        x = conv_block(x, 3, [256, 256, 1024], stage=4, block="a")
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block="b")
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block="c")

        x = GlobalAveragePooling2D(name="global_avg_pool")(x)
        predictions = Dense(outputs, activation="softmax")(x)

        self.model = Model(inputs=img_input, outputs=predictions)
        super().__init__(lr, decay)
