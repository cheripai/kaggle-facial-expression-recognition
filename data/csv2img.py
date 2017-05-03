import numpy as np
import os
import pandas as pd
import sys
from scipy.misc import imsave


TRAIN_PATH = "train"
VALID_PATH = "valid"
TEST_PATH = "test"
DATASET_PATHS = (TRAIN_PATH, VALID_PATH, TEST_PATH)


def pixelstr2array(pixelstr):
    img = np.array(pixelstr.split(" ")).astype(np.int32)
    img_size = int(np.sqrt(len(img)))
    return img.reshape((img_size, img_size))


if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])

    for dataset_path in DATASET_PATHS:
        if not os.path.exists(dataset_path):
            os.mkdir(dataset_path)

        for emotion in df["emotion"].unique():
            path_emotion = os.path.join(dataset_path, str(emotion))
            if not os.path.exists(path_emotion):
                os.mkdir(os.path.join(path_emotion))

    for i, row in df.iterrows():
        img = pixelstr2array(row["pixels"])
        fname = str(i)+".png"

        if row["Usage"] == "Training":
            path = os.path.join(TRAIN_PATH, str(row["emotion"]))
        elif row["Usage"] == "PublicTest":
            path = os.path.join(VALID_PATH, str(row["emotion"]))
        elif row["Usage"] == "PrivateTest":
            path = os.path.join(TEST_PATH, str(row["emotion"]))
        else:
            raise("Error: Invalid Usage: {}".format(row["Usage"]))

        imsave(os.path.join(path, fname), img)
