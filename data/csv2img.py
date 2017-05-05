import numpy as np
import os
import pandas as pd
import sys
from scipy.misc import imsave


TRAIN_PATH = "train"
VALID_PATH = "valid"
DATASET_PATHS = (TRAIN_PATH, VALID_PATH)
VALID_PROP = 0.10


if __name__ == "__main__":
    imgs = pd.read_csv(sys.argv[1], header=None).as_matrix()
    targets = pd.read_csv(sys.argv[2], header=None)[0]

    for dataset_path in DATASET_PATHS:
        if not os.path.exists(dataset_path):
            os.mkdir(dataset_path)

        for target in targets.unique():
            path_target = os.path.join(dataset_path, str(target))
            if not os.path.exists(path_target):
                os.mkdir(os.path.join(path_target))

    for i in range(len(targets)):
        img = imgs[i,:].reshape((48, 48))
        target = targets[i]
        fname = str(i)+".png"

        if i < VALID_PROP * len(targets):
            path = os.path.join(VALID_PATH, str(target))
        else:
            path = os.path.join(TRAIN_PATH, str(target))

        imsave(os.path.join(path, fname), img)
