import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge, LogisticRegression, LassoLars

# constantes
seed = 2021
np.random.seed(seed)
tf.random.set_seed(seed)
N_TRAIN = int(1e4)
BATCH_SIZE = 128
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE
data_steps = 4*6
np.seterr(divide='ignore')


class DutModel():
    def __init__(self, dev_id : str):
        self.model = None
        self.model_id = None
        self.dev_id = dev_id

    def load_model(self):
        pass

    def save_model(self):
        pass

    def predict(self):
        pass

    def train(self, training_data: pd.DataFrame) -> np.double:
        self.model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=(
            train_df.shape[0], train_df.shape[1]), return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(OUT_STEPS*num_features,
            kernel_initializer=tf.initializers.zeros()),
            tf.keras.layers.Reshape([OUT_STEPS, num_features])
        ])

        pass

    def save_dataset(self):
        pass

    def load_dataset(self):
        pass