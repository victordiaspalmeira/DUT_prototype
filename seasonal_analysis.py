#Análise sazonal para o RepeatBaseline

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge, LogisticRegression, LassoLars
from sklearn import preprocessing

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
    impute_data['Temperature'] = impute_data['Temperature'].rolling(20).sum() #moving average
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

if __name__ == "__main__":
    #Criação de dataframe
    dut_id = 'DUT209201107'
    filename = 'DUT209201107_20210301T000000_20210325T000000.csv'

    df = pd.read_csv(filename)
    df = clear_dataset(df)
    df = prepare_dataset(df)
    df.dropna(inplace=True)
    df = df['2021-03-02':'2021-03-12']

    df_decomposed = seasonal_decompose(df.values, period=24*6, model='additive')
    combine_seasonal_cols(df, df_decomposed)

    slope = pd.Series(np.gradient(df['trend'].values), df.index, name='slope')
    df['slope'] = 10*slope

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df.trend, mode='lines', name='Trend'))
    fig.add_trace(go.Scatter(x=df.index, y=df.slope, mode='lines', name='Slope'))
    fig.add_trace(go.Scatter(x=df.index, y=df.seasonal, mode='lines', name='Seasonality'))
    fig.add_trace(go.Scatter(x=df.index, y=df.residual, mode='lines', name='Residual'))
    fig.add_trace(go.Scatter(x=df.index, y=df.observed, mode='lines', name='Observed'))
    fig.add_trace(go.Scatter(x=df.index, y=len(df.index)*[0.27], mode='lines', name='Limit'))
    fig.update_layout(showlegend=True)
    fig.show(title='dut dado decomposto')
    fig = None
    print(df['trend'].mean())
    import plotly.express as px
    fig = px.histogram(df, x="trend")
    fig.show(title='histograma de temps')

