import pandas as pd
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge, LogisticRegression, LassoLars

import plotly.express as px
import plotly.graph_objects as go

def clear_dataset(df, data_steps=24*6):
    df.index.names = [None]
    df.set_index('timestamp', inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df.diff(periods=data_steps)
    df = df[~df.index.duplicated()]
    """
    if 'SSD' in df.columns:
        df.drop('SSD', inplace=True, axis=1)
    if 'COP' in df.columns:
        df.drop('COP', inplace=True, axis=1)
    if 'Pot_kw' in df.columns:
        df.drop('Pot_kw', inplace=True, axis=1)
    if 'Tsh' in df.columns:
        df.drop('Tsh', inplace=True, axis=1)
    if 'Tsc' in df.columns:
        df.drop('Tsc', inplace=True, axis=1)
    if 'dev_id' in df.columns:
        df.drop('dev_id', inplace=True, axis=1)
    """
    for col in df.columns:
      if col != 'Temperature' and col != 'timestamp':
        df.drop(col, inplace=True, axis=1)

    return df

def prepare_dataset(df):
  sampling = '10Min'
  df = df.resample(sampling).max()
  start_date = df.index[0]
  end_date = df.index[-1]

  cols = df.columns

  idx = pd.date_range(start=pd.Timestamp(start_date), end=pd.Timestamp(end_date), freq='10Min')
  df = df.reindex(idx)

  imputer = IterativeImputer(LogisticRegression())
  impute_data = pd.DataFrame(imputer.fit_transform(df))
  impute_data.columns = cols
  impute_data.index = df.index

  return impute_data

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=None, val_df=None, test_df=None,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])
    
  def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]

    if self.label_columns is not None:
      labels = tf.stack(
          [labels[:, :, self.column_indices[name]] for name in self.label_columns],
          axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels

  def find_anomalies(self, model=None, title='Plot', plot_col='Temperature'):
    print('Verificando anomalias...')
    inputs, labels = self.example
    plt.figure(figsize=(36, 24))
    plot_col_index = self.column_indices[plot_col]

    output_dict = dict()
    output_dict['Previs??o'] = list()
    output_dict['Real'] = list()
    output_dict['Labels'] = list()

    max_n = len(inputs) #min(max_subplots, len(inputs))

    anomaly_count = 0

    for n in range(max_n):
      inp = inputs[n, :, plot_col_index]
      output_dict['Real'].append(inp)

      if self.label_columns:
        label_col_index = self.label_columns_indices.get(plot_col, None)
      else:
        label_col_index = plot_col_index

      if label_col_index is None:
        continue

      lab = labels[n, :, label_col_index]
      output_dict['Labels'].append(lab)
      if model is not None:
        predictions = model(inputs)
        pred = predictions[n, :, label_col_index]
        output_dict['Previs??o'].append(pred)

      real = np.roll(output_dict['Real'][n], 3)
      labels = np.roll(output_dict['Labels'][n], 3)
      preds = np.roll(output_dict['Previs??o'][n], 3)

      abs_error = abs(labels - preds)
      anomalies = abs_error > 2*np.std(abs_error)
      anomaly_count += anomalies.sum()
      
      if n == 0:
        pass

    print("Poss??veis anomalias:", anomaly_count)
    return anomaly_count

def plot(self, model=None, title='Plot', plot_col='Temperature', max_subplots=3,):
  inputs, labels = self.example
  plt.figure(figsize=(36, 24))
  plot_col_index = self.column_indices[plot_col]

  output_dict = dict()
  output_dict['Previs??o'] = list()
  output_dict['Real'] = list()
  output_dict['Labels'] = list()

  max_n = min(max_subplots, len(inputs))
  for n in range(max_n):
    plt.subplot(max_n, 1, n+1)
    plt.ylabel(f'{plot_col} [normed]')
    inp = inputs[n, :, plot_col_index]
    plt.plot(self.input_indices, inp,
             label='Inputs', marker='.', zorder=-10)

    output_dict['Real'].append(inp)

    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index

    if label_col_index is None:
      continue
    lab = labels[n, :, label_col_index]
    output_dict['Labels'].append(lab)
    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
      predictions = model(inputs)
      pred = predictions[n, :, label_col_index]
      output_dict['Previs??o'].append(pred)

      plt.scatter(self.label_indices, pred,
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)

      plt.scatter(self.label_indices, abs(np.roll(predictions[n, :, label_col_index], 6) - np.roll(labels[n, :, label_col_index], 6)),
                  marker='X', edgecolors='k', label='Error',
                  c='#ff1f0e', s=64)

    if n == 0:
      plt.legend()

  plt.title(title)
  plt.xlabel('Time [10 Min]')
  plt.savefig('{}_plot.png'.format(title))

WindowGenerator.plot = plot

def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)

  split = self.split_window
  ds = ds.map(split)

  return ds

WindowGenerator.make_dataset = make_dataset

@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.test))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example