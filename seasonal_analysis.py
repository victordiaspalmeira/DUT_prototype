#Análise sazonal para o RepeatBaseline
import datetime
from numpy.lib import real
import pandas as pd
import numpy as np
from scipy.signal import wavelets
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge, LogisticRegression, LassoLars
from sklearn import preprocessing

import scipy.fft  
import scipy.signal
import plotly.express as px
import plotly.graph_objects as go

from statsmodels.tsa.seasonal import seasonal_decompose


#constantes
np.random.seed(2021)

def clear_dataset(df):
    df.index.names = [None]

    df.set_index('timestamp', inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated()]

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

    return df

def prepare_dataset(df):
    #sampling
    sampling = '10Min'
    df = df.resample(sampling).mean()

    cols = df.columns

    #criando indexes faltantes
    start_date = df.index[0]
    end_date = df.index[-1]
    idx = pd.date_range(start=pd.Timestamp(start_date), end=pd.Timestamp(end_date), freq='10Min')
    df = df.reindex(idx)

    #imputação
    imputer = IterativeImputer(LogisticRegression())
    impute_data = pd.DataFrame(imputer.fit_transform(df))
    impute_data.columns = cols
    impute_data.index = df.index

    #smoothing
    #impute_data['Temperature'] = impute_data['Temperature'].rolling(20).sum() #moving average
    impute_data['Temperature'] = preprocessing.minmax_scale(impute_data['Temperature'])
    return impute_data

def combine_seasonal_cols(input_df, seasonal_model_results):
    """Adds inplace new seasonal cols to df given seasonal results

    Args:
        input_df (pandas dataframe)
        seasonal_model_results (statsmodels DecomposeResult object)
    """
    # Add results to original df
    input_df['observed'] = seasonal_model_results.observed
    input_df['residual'] = seasonal_model_results.resid
    input_df['seasonal'] = seasonal_model_results.seasonal
    input_df['trend'] = seasonal_model_results.trend


def data_fft(input_df : pd.DataFrame):
    output = pd.DataFrame()
    for key in ['observed', 'residual', 'seasonal', 'trend']:
        #fourier = input_df[key].rolling(64).map(np.fft.fft)
        fourier = np.fft.fft(input_df[key])
        output[f"{key}_fft_r"] = np.real(fourier)
        output[f"{key}_fft_i"] = np.imag(fourier)
        #output[f'{key}_dct'] = scipy.fft.dct(input_df[key])
        
    return output

def data_stft(input_df : pd.DataFrame):
    w = 6.
    fs = 1 / (input_df.index[1] - input_df.index[0]).total_seconds()
    output = pd.DataFrame()
    output['observed'] = input_df['observed']
    output['trend'] = input_df['trend']
    output['residual'] = input_df['residual']

    freqs, times, stft = scipy.signal.stft(input_df['observed'], fs = fs)

    print(stft)
    output['stft_observed'] = stft

    return output

def data_wavelet(input_df : pd.DataFrame):
    w = 4.
    fs = 1 / (input_df.index[1] - input_df.index[0]).total_seconds() # sampling frequency is constant - data was resampled
    output = pd.DataFrame()

    freq = np.linspace(0, fs, 256)
    widths = w*fs / (2*freq*np.pi)

    output['observed'] = input_df['observed']
    output['trend'] = input_df['trend']
    output['residual'] = input_df['residual']
    wavelet = scipy.signal.cwt(input_df['observed'], scipy.signal.morlet2, widths)
    # output['observed_wavelet_sum_r'] = np.zeros((len(input_df['observed']),))
    # output['observed_wavelet_sum_i'] = np.zeros((len(input_df['observed']),))
    for i, wavelet_out in enumerate(wavelet):
        
        output[f'obs_{i}'] = np.real(wavelet_out)
        #output[f'observed_wavelet_{i}_phase'] = np.angle(wavelet_out)
        # output['observed_wavelet_sum_r'] += output[f'observed_wavelet_{i}_r']
        # output['observed_wavelet_sum_i'] += output[f'observed_wavelet_{i}_i']
        #output['observed_wavelet_sum'] += wavelet_out
    wavelet = scipy.signal.cwt(input_df['trend'], scipy.signal.ricker, widths)
    # output['trend_wavelet_sum_r'] = np.zeros((len(input_df['trend']),))
    # output['trend_wavelet_sum_i'] = np.zeros((len(input_df['trend']),))
    
    for i, wavelet_out in enumerate(wavelet):
        output[f'trend_{i}'] = np.real(wavelet_out)
        #output[f'trend_wavelet_{i}_phase'] = np.ang(wavelet_out)
        # output['trend_wavelet_sum_r'] += output[f'trend_wavelet_{i}_r']
        # output['trend_wavelet_sum_i'] += output[f'trend_wavelet_{i}_i']
    


    wavelet = scipy.signal.cwt(input_df['residual'], scipy.signal.morlet2, widths)
    # output['residual_wavelet_sum_r'] = np.zeros((len(input_df['residual']),))
    # output['residual_wavelet_sum_i'] = np.zeros((len(input_df['residual']),))
    
    for i, wavelet_out in enumerate(wavelet):
        output[f'residual_{i}'] = np.real(wavelet_out)
        #output[f'residual_wavelet_{i}_i'] = np.imag(wavelet_out)
        # output['residual_wavelet_sum_r'] += output[f'residual_wavelet_{i}_r']
        # output['residual_wavelet_sum_i'] += output[f'residual_wavelet_{i}_i']

    return output

def date_iterator_period(start_date : datetime.datetime, end_date : datetime.datetime, period : datetime.timedelta):
    current_date = start_date
    while current_date + period < end_date:
        yield current_date, current_date + period
        current_date += period
    yield current_date, end_date


if __name__ == "__main__":
    #Criação de dataframe
    dut_id = 'DUT209201107'
    filename = 'DUT209201107_20210301T000000_20210325T000000.csv'

    df = pd.read_csv(filename)
    df = clear_dataset(df)
    df = prepare_dataset(df)
    df.dropna(inplace=True)
    time_increment = datetime.timedelta(days=3)
    start_date = datetime.datetime(year=2021, month=3, day=14)
    end_date = datetime.datetime(year=2021, month=3, day=23)
    
    dfs = [df[start : end] for start, end in date_iterator_period(start_date, end_date, time_increment)]

    df = df['2021-03-10':'2021-03-16']
    df['Temperature'] = preprocessing.MinMaxScaler(feature_range=(-1,1)).fit_transform(df)
    print(df)
    #for df in dfs:
    df_decomposed = seasonal_decompose(df.values, period=24*6, model='additive')
    combine_seasonal_cols(df, df_decomposed)
    df.dropna(inplace=True)
    slope = pd.Series(np.gradient(df['trend'].values), df.index, name='slope')
    df['slope'] = 10*slope
    ffts = data_fft(df)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=df.index, y=df.trend, mode='lines', name='Trend'))
    fig.add_trace(go.Scattergl(x=df.index, y=df.slope, mode='lines', name='Slope'))
    fig.add_trace(go.Scattergl(x=df.index, y=df.seasonal, mode='lines', name='Seasonality'))
    fig.add_trace(go.Scattergl(x=df.index, y=df.residual, mode='lines', name='Residual'))
    fig.add_trace(go.Scattergl(x=df.index, y=df.observed, mode='lines', name='Observed'))
    fig.add_trace(go.Scattergl(x=df.index, y=len(df.index)*[0.27], mode='lines', name='Limit'))

    fig.update_layout(showlegend=True)
    fig.show(title=f'dut dado decomposto {df.index[0]} a {df.index[-1]}')
    fig = None
        
    
    # fig2 = go.Figure()
    # for column in ffts.columns:
    #     fig2.add_trace(go.Scattergl(x=ffts.index, y=ffts[column], mode='lines', name=column))
    # fig2.update_layout(showlegend=True)
    # fig2.show(title=f'DFT / DCT / Wavelet (Rickie wavelet) transform {df.index[0]} a {df.index[-1]}')
    # fig2 = None

    df.dropna(inplace=True)
    wavelet = data_wavelet(df)
    df.dropna(inplace=True)

    fig2 = go.Figure()
    for column in wavelet.columns:
        fig2.add_trace(go.Scattergl(x=wavelet.index, y=wavelet[column], mode='lines', name=column))
    fig2.update_layout(showlegend=True)
    fig2.show(title=f'DFT / DCT / Wavelet (Rickie wavelet) transform {df.index[0]} a {df.index[-1]}')
    fig2 = None

    # stft = data_stft(df)
    # fig2 = go.Figure()
    # for column in stft.columns:
    #     fig2.add_trace(go.Scattergl(x=stft.index, y=stft[column], mode='lines', name=column))
    # fig2.update_layout(showlegend=True)
    # fig2.show(title=f'DFT / DCT / Wavelet (Rickie wavelet) transform {df.index[0]} a {df.index[-1]}')
    # fig2 = None

        # import plotly.express as px
        # fig = px.histogram(df, x="trend")
        # fig.show(title='histograma de temps')

