import pandas as pd
import numpy as np

import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import tensorflow as tf
sys.stderr = stderr

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.metrics import *

def get_model(x_train):
    """
    Return the requested Keras model

    :param x_train:      The NumPy array used for training (for dimension size extraction)


    :return:             The Keras model
    """
    
    # some parameters to control model
    dp_lvl = 0.2
    regularizer_lvl = 0.002
    
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # network design
        model = Sequential()
        <<<TODO>>> # create your model here
    
    return model
