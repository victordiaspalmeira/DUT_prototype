from datetime import datetime, timedelta
import os
from socket import create_connection
from query_intel import dut_query
from typing import Optional
from mysql.connector.cursor import CursorBase, MySQLCursorBufferedDict
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

from seasonal_analysis import prepare_dataset

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge, LogisticRegression, LassoLars
from data_processing import *

#import pickle as p
import sql_handler
import s3_handler
import zipfile
import pickle

from contextlib import closing


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


class DutModel:
    model_bucket = 'intel-model-bucket'
    sampling_rate = timedelta(minutes=10)
    def __init__(self, dev_id: str):
        self.model: Optional[tf.keras.Sequential] = None
        self.scaler = None
        self.model_id: Optional[int] = None
        self.dev_id = dev_id
        self.dataset: Optional[pd.DataFrame] = None

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        
        with closing(sql_handler.create_connection()) as db:
            with closing(db.cursor(buffered=True, dictionary=True)) as cursor:
                assert isinstance(cursor, MySQLCursorBufferedDict)
                cursor.execute('select * from duts where dev_id=%s', (self.dev_id,))
                if len(cursor.fetchall()) == 0:
                    cursor.execute('insert into duts (dev_id) values (%s)', (self.dev_id,))
                    db.commit()       

    @staticmethod
    def from_sql_db(dev_id: str, model_id: Optional[int] = None):
        """Loads a model with data in the local DB and returns it. Model must be from id dev_id. If model_id is None, will pick the first 'Active' model.

        Args:
            dev_id (str): Device ID.
            model_id (Optional[int], optional): ID of the model to be retrieved. Defaults to None.

        Returns:
            DutModel: Instantiated DutModel with model already loaded. 
        """
        if model_id is None:
            sql = 'select * from duts inner join dutmodels on duts.model_id == dutmodels.ID where duts.dev_id=%s'
        else:
            sql = 'select * from dutmodels where dev_id=%s and model_id=%s'
        params = [dev_id] + ([model_id] if model_id is not None else [])
        with closing(sql_handler.create_connection()) as db:
            with closing(db.cursor(buffered=True, dictionary=True)) as cursor:
                assert isinstance(cursor, MySQLCursorBufferedDict)
                cursor.execute(sql, params)
                model_data = cursor.fetchone()

        obj = DutModel(model_data['dev_id'])
        obj.load_model(model_id)
        return obj

    @property
    def model_path(self):
        return f"./models/{self.dev_id}"

    @property
    def dataset_path(self):
        return f"./datasets/{self.dev_id}"

    @property
    def scaler_name(self):
        return f"scaler_{self.dev_id}.p"

    def load_model(self, model_id: int):
        """Loads a model from s3.

        Args:
            model_id (int): Model ID to be queried from the local DB

        Raises:
            ValueError: On invalid model_id (non integer or nonpositive)
        """
        if not isinstance(model_id, int):
            raise ValueError('Model ID needs to be an integer')

        if model_id < 0:
            raise ValueError('Model ID must be positive')

        zip_path = s3_handler.download_from_s3(
            'intel-model-bucket', f"{self.dev_id}_{model_id}.zip")

        with zipfile.ZipFile(zip_path, 'r') as file:
            file.extractall(self.model_path)

        self.model = tf.keras.models.load_model(self.model_path)

        with open(self.model_path + '/' + self.scaler_name, "rb") as scaler:
            self.scaler = pickle.load(scaler)

    def save_model(self):
        if not self.model or not self.scaler or not self.model_id:
            raise ValueError('No model is loaded to be saved!')

        self.model.save(self.model_path)
        filepath = f'./tmp/{self.dev_id}.zip'

        with open(self.model_path + '/' + self.scaler_name, 'wb') as f:
            pickle.dump(self.scaler, f)

        try:
            zf = zipfile.ZipFile(filepath, 'x')
        except FileExistsError:
            os.remove(filepath)
            zf = zipfile.ZipFile(filepath, 'x')

        cwd = os.getcwd()
        os.chdir(self.model_path)
        for root, dirs, files in os.walk('.'):
            for d in dirs:
                zf.write(os.path.join(root, d))
            for f in files:
                zf.write(os.path.join(root, f))
        zf.close()
        os.chdir(cwd)

        s3_handler.upload_to_s3(DutModel.model_bucket,
                                filepath, f'{self.dev_id}_{self.model_id}.zip')

    def predict(self, input_data: pd.DataFrame, title='Default'):
        # Carrega scaler
        filename = self.model_path + '/' + self.scaler_name
        infile = open(filename, 'rb')
        scaler = pickle.load(infile)

        input_data = (input_data - scaler['mean'])/scaler['std']

        # Criar janela para previsão
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
        """Trains a model for a DUT and saves it to the DB.

        Args:
            training_data (pd.DataFrame): DataFrame containing data, to be split into training data and testing data

        Returns:
            np.double: Model evaluate statistic.
        """
        # Split datasets
        n = len(training_data)
        train_df = training_data[0:int(n*0.7)]
        val_df = training_data[int(n*0.7):int(n*0.9)]
        test_df = training_data[int(n*0.9):]

        num_features = training_data.shape[1]

        # Standarize
        self.scaler = dict()
        self.scaler['mean'] = train_df.mean()
        self.scaler['std'] = train_df.std()
        # Salvando em pickle (provisório)
        filename = self.model_path + '/' + self.scaler_name
        outfile = open(filename, 'wb')
        pickle.dump(self.scaler, outfile)
        outfile.close()

        # Normalização
        train_df = (train_df - self.scaler['mean'])/self.scaler['std']
        val_df = (val_df - self.scaler['mean'])/self.scaler['std']
        test_df = (test_df - self.scaler['mean'])/self.scaler['std']

        # Criação de janela de dados
        window = WindowGenerator(input_width=data_steps,
                                 label_width=data_steps,
                                 shift=data_steps,
                                 train_df=train_df, val_df=val_df, test_df=test_df
                                 )

        window.plot(self.model, title='FIT')

        # Modelo
        self.model = tf.keras.Sequential([
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                64, input_shape=(train_df.shape[0], train_df.shape[1]))),
            #tf.keras.layers.LSTM(64, activation='relu', return_sequences=True),
            # tf.keras.layers.Dropout(0.3),
            #tf.keras.layers.LSTM(64, activation='relu'),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(60, activation="relu"),
            tf.keras.layers.Dense(20, activation="relu"),
            tf.keras.layers.Dense(OUT_STEPS*num_features,
                                  kernel_initializer=tf.initializers.zeros()),
            tf.keras.layers.Reshape([OUT_STEPS, num_features])
        ])

        # learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            0.001,
            decay_steps=STEPS_PER_EPOCH*1000,
            decay_rate=1,
            staircase=False
        )

        # early stopping. restaura pesos pra a melhor config
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

        evaluate = 0.1 #colocar evaluate
        

        self.__save_model_to_sql(evaluate, training_data.index[0].to_pydatetime(), training_data.index[-1].to_pydatetime())

        return history


    def __save_model_to_sql(self, evaluate, start_timestamp, end_timestamp):
        query = 'insert into dutModels (dev_id, evaluate, train_timestamp, start_timestamp, end_timestamp, model_state) values (%s, %s, %s, %s, %s)'
        
        with closing(sql_handler.create_connection()) as db:
            with closing(db.cursor(buffered=True, dictionary=True)) as cursor:
                assert isinstance(cursor, MySQLCursorBufferedDict)
                parameters = [self.dev_id, evaluate, datetime.now(), start_timestamp, end_timestamp]
                cursor.execute(query, parameters)
                
                query = 'select MAX(ID) from dutModels where dev_id=%s'
                cursor.execute(query, [self.dev_id])
                self.model_id = cursor.fetchone()['MAX(ID)']
                
                cursor.execute('update duts set model_id=%s where dev_id=%s', (self.model_id, self.dev_id))
                db.commit()




    def save_dataset(self, path: Optional[str] = None) -> str:
        """Saves a dataset in the path passed to it as a pickled object.
 
        Args:
            path (Optional[str], optional): Path as string. Defaults to None.

        Raises:
            ValueError: If dataset is None.

        Returns:
            str: Path where the dataset was saved.
        """
        if self.dataset is None:
            raise ValueError("Dataset is None")
        path = path or f"{self.dataset_path}/curr_dataset.p"
        self.dataset.to_pickle(path)
        return path

    def load_dataset(self, start_time: datetime, end_time: datetime, path=None):
        """Loads a dataset from start_time to end_time, first looking at the default pickled path and queries the rest from DynamoDB.

        Args:
            start_time (datetime): Start time of the dataset.
            end_time (datetime): End time of the dataset.

        Raises:
            ValueError: If parameters are not datetime.datetime
            ValueError: If end_time < start_time
        """

        if not isinstance(start_time, datetime) or not isinstance(end_time, datetime):
            raise ValueError("start_time and end_time should be datetimes")

        if end_time < start_time:
            raise ValueError("end_time occurs before start_time")

        path = path or f"{self.dataset_path}/curr_dataset.p"

        try:
            self.dataset: pd.DataFrame = pd.read_pickle(path)
            self.dataset.drop(
                index=self.dataset.loc[self.dataset.index < start_time], inplace=True)
            self.dataset.drop(
                index=self.dataset.loc[self.dataset.index > end_time], inplace=True)
            self.dataset.sort_index(inplace=True)
        except Exception as e:
            self.dataset : pd.DataFrame = dut_query(self.dev_id, start_time, end_time)
            self.dataset = prepare_dataset(self.dataset)
            return

        # Assumindo que o dataset é um bloco único (em acordo com nossas práticas até agora),
        # há 3 hipóteses: dataset já existente compreende parte (até totalidade) do novo período a partir do início, a partir do fim e dividindo no meio

        # Ou seja, sendo '+' dados existentes localmente e '=' dados não existentes:
        # +++==== Caso 1 - start_time = fim do dataframe, end_time=end_time
        # ==+++== Caso 2 - 2 queries, start_time1 = start_time, end_time1=inicio do dataframe e start_time2=fim do dataframe, end_time2=end_time
        # ====+++ Caso 3 - start_time=start_time, end_time=inicio do dataframe

        # No caso +++++++, a diferença entre start_time e end_time será 0

        if start_time == self.dataset.index[0]:  # caso 1
            start_time = self.dataset.index[-1]
            self.dataset = pd.concat(
                [self.dataset, dut_query(self.dev_id, start_time, end_time)])
        elif end_time == self.dataset.index[-1]:  # caso 3
            end_time = self.dataset.index[0]
            self.dataset = pd.concat(
                [dut_query(self.dev_id, start_time, end_time), self.dataset])
        else:  # caso 2
            start_time1, end_time1 = start_time, self.dataset.index[0]
            start_time2, end_time2 = self.dataset.index[-1], end_time
            self.dataset = pd.concat([dut_query(self.dev_id, start_time1, end_time1), self.dataset, dut_query(
                self.dev_id, start_time2, end_time2)])
        
        self.dataset = prepare_dataset(self.dataset)
        print(self.dataset)

if __name__ == '__main__':
    dev_id = 'DUT209201107'

    dut = DutModel(dev_id)
    dut.load_dataset(datetime(2021, 3, 1), datetime(2021, 3, 5))
    dut.save_dataset()
    dut.train(dut.dataset)
    dut.save_model()
    exit()
    df_train = pd.read_csv('DUT209201107_training.csv')
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