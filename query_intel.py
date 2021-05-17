# -*- coding: utf-8 -*-
import datetime
#from intel_info_connect import get_devPeriod, get_devPeriods_data, insert_devPeriods_data, insert_predictLog
from typing import Union

import numpy as np
import pandas as pd

import dynamo_querier
import traceback

from api_server_interface import api_conn

ewma = pd.Series.ewm

import dotenv
dotenv.load_dotenv("./.env")

def dynamodb_query(dac_id : str, start_time : Union[str, datetime.datetime], end_time : Union[str, datetime.datetime], attributes_to_get=None):
    """Realiza um query no DynamoDB na tabela RAW e retorna um dataframe."""


    if isinstance(start_time, str):
        start_time = start_time
    else:
        start_time = start_time.isoformat()
    
    if isinstance(end_time, str):
        end_time = end_time
    else:
        end_time = end_time.isoformat()

    query_data = dynamo_querier.get_query_data(dac_id, start_time, end_time, attributes_to_get)

    return dynamo_querier.dynamo_querier.query_all_proc(query_data).reset_index(drop=True)

def dut_query(dev_id : str, start_time : datetime.datetime, end_time : datetime.datetime, save=False) -> pd.DataFrame:
    associated_dacs = sorted(api_conn.get_associated_dacs(dev_id)['list'], key=lambda x : x['DAC_ID'])
    dut_data = dynamodb_query(dev_id, start_time, end_time, attributes_to_get=['timestamp', 'Temperature'])
    dut_data['timestamp'] =  pd.to_datetime(dut_data['timestamp'])             
    dut_data.drop(index = dut_data.loc[dut_data['timestamp'] < start_time].index,inplace = True)
    dut_data.set_index('timestamp',inplace = True)
   
    #dut_data = dut_data[~dut_data.index.duplicated(keep='last')]
    #print(dut_data[dut_data.index.duplicated()])
        
    

    for i, dac in enumerate(associated_dacs):
        dac_data = dynamodb_query(dac['DAC_ID'], start_time, end_time, attributes_to_get=["timestamp", "L1", "T0"])
        
        dac_data['timestamp'] =  pd.to_datetime(dac_data['timestamp'])             
        
        dac_data.drop(index = dac_data.loc[dac_data['timestamp'] < start_time].index,inplace = True)
        #dac_data = dac_data.drop_duplicates(keep='first')
        dac_data.set_index('timestamp',inplace = True)
        dac_data = dac_data[~dac_data.index.duplicated(keep='last')]
        #print(dac_data[dac_data.duplicated()])     
        dac_data = dac_data.reindex(dut_data.index, method='nearest') # setting them both to same index
        
        #print(dac_data)
        dac_data = dac_data.add_suffix(f'_{i}') #adding _i to the end of the column names
        dut_data = pd.concat([dut_data, dac_data], axis='columns')

    if save:
        dut_data.to_csv(f'./{dev_id}_{start_time.isoformat().replace(":", "").replace(" ", "").replace("-", "")}_{end_time.isoformat().replace(":", "").replace(" ", "").replace("-", "")}.csv')

    return dut_data


if __name__ == "__main__":
    dev_ids = ['DUT209201107']
    #print(fetch_dac_data(dev_id))
    start_time = datetime.datetime(2021,5,1)
    #end_time = datetime.datetime(2021,3,25)
    #query(dev_id, start_time, end_time, save=False)
    now = datetime.datetime.now()
    #print(f"Starting query at {now}")
    #print(query(dev_id, now - datetime.timedelta(hours=1), now, save=False))
    #query(dev_id,start_time='2020-11-05', end_time='2021-04-01', period_precision=datetime.timedelta(days=21), save=True)
    for dev_id in dev_ids:
        dut_data = dut_query(dev_id, start_time, now, save=True)
    print(f"Total : {(datetime.datetime.now() - now).total_seconds()} s")
    print(dut_data)
    
    #query_manual()
