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
from data_processing import *

# constantes
seed = 2021
np.random.seed(seed)
tf.random.set_seed(seed)
N_TRAIN = int(1e4)
BATCH_SIZE = 128
MAX_EPOCHS = 3000
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE
data_steps = 4*6
OUT_STEPS = data_steps
np.seterr(divide='ignore')
patience = 350

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
        """
        Normalização e limpeza dos dados
        """

        #Split datasets
        n = len(df)
        train_df = training_data[0:int(n*0.7)]
        val_df = training_data[int(n*0.7):int(n*0.9)]
        test_df = training_data[int(n*0.9):]

        num_features = training_data.shape[1]

        #Standarize
        train_mean = train_df.mean()
        train_std = train_df.std()

        train_df = (train_df - train_mean)/train_std
        val_df = (val_df - train_mean)/train_std
        test_df = (test_df - train_mean)/train_std

        """
        Criação de janela de dados
        """
        window = WindowGenerator(input_width=data_steps,
                                    label_width=data_steps,
                                    shift=data_steps,
                                    train_df=train_df, val_df=val_df, test_df=test_df
                                    )       

        window.plot()

        """
        Modelo
        """
        #Modelo
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

        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            0.01,
            decay_steps=STEPS_PER_EPOCH*1000,
            decay_rate=1,
            staircase=False
        )        

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                            patience=patience,
                                                            mode='auto',
                                                            restore_best_weights=True)

        self.model.compile(loss=tf.losses.MeanSquaredError(),
                            optimizer=tf.optimizers.Adam(lr_schedule),
                            metrics=[tf.metrics.MeanAbsoluteError()])        

        history = self.model.fit(window.train, epochs=MAX_EPOCHS,
                            validation_data=window.val,
                            callbacks=[early_stopping])

        return history

    def save_dataset(self):
        pass

    def load_dataset(self):
        pass

if __name__ == '__main__':
    dev_id = 'DUT209201107'
    df = pd.read_csv('DUT209201107_20210301T000000_20210325T000000.csv')

    df = clear_dataset(df)
    df = prepare_dataset(df)
    df = df['2021-03-03':'2021-03-12']

    dutModel = DutModel(dev_id=dev_id)
    history = dutModel.train(training_data=df)

