from dut_model import DutModel
import pandas
import numpy
import datetime
import pytest
"""
Modelos -> Armazenados no S3, encapsulados numa classe
Dut_Model -> Representa um dut - funcionalidades 

"""


def create_default_model(model_id) -> DutModel:
    dev_id = 'test_id'
    dut = DutModel(dev_id=dev_id)
    dut.load_model(model_id=model_id)
    return dut


def test_fails_bad_model_id():
    model_ids = ['jlgjlskhgbnosdflk', -1, float('NaN'), float('Inf'), 1.]
    for model_id in model_ids:
        with pytest.raises(ValueError):
            create_default_model(model_id)


def test_config_model():
    dev_id = 'test_id'
    dut = DutModel(dev_id=dev_id)
    dut.load_model(model_id=12)
    assert dut.model.model_id == 12

def test_load_dataset():
    dut = create_default_model(1)
    start_time = datetime.datetime(2020, 9, 10)
    end_time = datetime.datetime(2020, 10, 10)
    dut.load_dataset(start_time=start_time, end_time=end_time)
    assert isinstance(dut.dataset, pandas.DataFrame)
    assert isinstance(dut.dataset.index, pandas.DatetimeIndex)
    assert 'Temperature' in dut.dataset.columns
    assert dut.dataset.index[0] >= pandas.to_datetime(start_time)
    assert dut.dataset.index[-1] <= pandas.to_datetime(end_time)

def test_bad_load_dataset():
    start_time = datetime.datetime(2020, 10, 10)
    end_time = datetime.datetime(2020, 9, 10)

    dut = create_default_model(1)
    with pytest.raises(ValueError):
        dut.load_dataset(start_time, end_time)

    with pytest.raises(ValueError):
        dut.load_dataset("asfdffd", "sdhgf")

def test_save_dataset():
    start_time = datetime.datetime(2020, 9, 10)
    end_time = datetime.datetime(2020, 10, 10)
    dut = create_default_model(1)
    dut.load_dataset(start_time=start_time, end_time=end_time)
    path = dut.save_dataset()
    assert isinstance(path, str)

def test_bad_save_dataset():
    dut = create_default_model(1)
    with pytest.raises(ValueError):
        dut.save_dataset()

def test_predict():
    dev_id = 'test_id'
    dut = DutModel(dev_id=dev_id)
    dut.load_model(model_id=1)
    dut.load_dataset(start_time='2020-09-10', end_time='2020-10-10')
    failure_data = dut.predict()
    assert isinstance(failure_data, numpy.ndarray)

def test_train():
    pass