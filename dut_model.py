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

import pickle as p
# constantes
seed = 2021
np.random.seed(seed)
tf.random.set_seed(seed)
N_TRAIN = int(1e4)
BATCH_SIZE = 64
MAX_EPOCHS = 2500
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE
data_steps = 24*6
OUT_STEPS = data_steps
np.seterr(divide='ignore')
patience = 50

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

class DutModel():
    def __init__(self, dev_id : str):
        self.model = None
        self.model_id = None
        self.dev_id = dev_id

    def load_model(self):
        pass

    def save_model(self):
        pass

    def predict(self, input_data: pd.DataFrame, title='Default'):
        #Carrega scaler
        filename = 'scaler_{}.p'.format(self.dev_id)
        infile = open(filename, 'rb')
        scaler = p.load(infile)

        input_data = (input_data - scaler['mean'])/scaler['std']

        #Criar janela para previsão
        window = WindowGenerator(input_width=data_steps,
                                    label_width=data_steps,
                                    shift=data_steps,
                                    train_df=input_data,
                                    val_df=input_data,
                                    test_df=input_data
                                )       

        hist = self.model.predict(window.test)
        window.plot(self.model, title=title)
        return hist

    def train(self, training_data: pd.DataFrame) -> np.double:
        #Split datasets
        n = len(training_data)
        train_df = training_data[0:int(n*0.7)]
        val_df = training_data[int(n*0.7):int(n*0.9)]
        test_df = training_data[int(n*0.9):]

        num_features = training_data.shape[1]

        #Standarize
        scaler = dict()
        scaler['mean'] = train_df.mean()
        scaler['std'] = train_df.std()

        #Salvando em pickle (provisório)
        filename = 'scaler_{}.p'.format(self.dev_id)
        outfile = open(filename, 'wb')
        p.dump(scaler, outfile)
        outfile.close()

        #Normalização
        train_df = (train_df - scaler['mean'])/scaler['std']
        val_df = (val_df - scaler['mean'])/scaler['std']
        test_df = (test_df - scaler['mean'])/scaler['std']

        #Criação de janela de dados
        window = WindowGenerator(input_width=data_steps,
                                    label_width=data_steps,
                                    shift=data_steps,
                                    train_df=train_df, val_df=val_df, test_df=test_df
                                )       

        window.plot(self.model, title='FIT')

        #Modelo
        self.model = tf.keras.Sequential([
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, input_shape=(train_df.shape[0], train_df.shape[1]))),
            #tf.keras.layers.LSTM(64, activation='relu', return_sequences=True),
            #tf.keras.layers.Dropout(0.3),
            #tf.keras.layers.LSTM(64, activation='relu'),
            #tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(60, activation="relu"),
            tf.keras.layers.Dense(20, activation="relu"),
            tf.keras.layers.Dense(OUT_STEPS*num_features,
            kernel_initializer=tf.initializers.zeros()),
            tf.keras.layers.Reshape([OUT_STEPS, num_features])
        ])

        #learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            0.001,
            decay_steps=STEPS_PER_EPOCH*1000,
            decay_rate=1,
            staircase=False
        )        

        #early stopping. restaura pesos pra a melhor config
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                            patience=patience,
                                                            mode='auto',
                                                            restore_best_weights=True)

        self.model.compile(loss=tf.losses.Huber(),
                            optimizer=tf.optimizers.Adam(lr_schedule))        

        history = self.model.fit(window.train, epochs=MAX_EPOCHS,
                            validation_data=window.val,
                            validation_steps=80,
                            callbacks=[early_stopping])

        return history

    def save_dataset(self):
        pass

    def load_dataset(self):
        pass

if __name__ == '__main__':
    dev_id = 'DUT209201107'

    #df_train = pd.read_csv('DUT209201107_training.csv')

    #df_train = pd.read_csv('DUT209201120_train.csv')
    df_train = pd.read_csv('DUT209201153_train.csv').rolling(3).mean()
    df_train = clear_dataset(df_train)
    df_train = prepare_dataset(df_train)
    print(df_train)

    #df_train = df_train['2021-03-03':'2021-03-12']
    #df_test = pd.read_csv('DUT209201107_maio.csv')
    #df_test = pd.read_csv('DUT209201120_train.csv')
    #df_test = clear_dataset(df_test)
    #df_test = prepare_dataset(df_test)
    #print(df_test.columns)

    dutModel = DutModel(dev_id=dev_id)
    history_train = dutModel.train(training_data=df_train)
    #history_predict_1 = dutModel.predict(input_data=df_test, title='TEST')
    history_predict_2 = dutModel.predict(input_data=df_train, title='TRAIN')
