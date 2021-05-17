import datetime
import boto3
import pandas

from boto3.dynamodb.types import TypeDeserializer
from functools import partial

from typing import Iterable

class CustomDeserializer(TypeDeserializer): #custom DynamoDB deserializer that doesn't use Decimal
    def __init__(self) -> None:
        super().__init__()

    def _deserialize_n(self, value):
        '''Deserializes a number into either a float or an int - gets called by super().deserialize()
        '''
        number = float(value)
        if number.is_integer():
            number = int(number)
        return number

def create_timestamps(data_dict):
    '''Turns the timestamp field for every data pack into a list containing the timestamps for every individual sample, based on the only timestamp sent
    '''
    if 'L1' in data_dict.keys():
        max_len = len(data_dict['L1'])
        data_period = 1
    elif 'Temperature' in data_dict.keys():
        max_len = len(data_dict['Temperature'])
        data_period = 5
    else:
        raise TypeError
    try:
        base_time = datetime.datetime.fromisoformat(data_dict['timestamp'])
        data_dict['timestamp'] = list([base_time - k*datetime.timedelta(seconds=data_period) for k in range(max_len - 1, -1, -1)])
        #print(f"{len(data_dict['timestamp'])} | {data_dict['timestamp']}")
    except ValueError: #there is no timestamp, just ignore it
        pass
    return data_dict

class dynamo_querier:
    dynamo = boto3.client('dynamodb')
    
    @classmethod
    def query_single_table(cls, table_name, **kwargs):
        ignored_fields = ['bt_id', 'MAC', 'package_id']
        paginator = cls.dynamo.get_paginator('query')

        result = pandas.DataFrame()
        i = 1
        deserialiser = CustomDeserializer()
        for page in paginator.paginate(TableName=table_name, **kwargs):
            items = map(lambda x : create_timestamps({k : deserialiser.deserialize(v) for k,v in x.items() if k not in ignored_fields}), page['Items'])
            result = pandas.concat([result, *map(pandas.DataFrame.from_records, items)])
            i += 1
        return result

    @classmethod
    def query_all_proc(cls,kwargs):
        tables = []
        results = pandas.DataFrame()
        paginator = cls.dynamo.get_paginator("list_tables")
        from_records = partial(pandas.DataFrame.from_records, index="timestamp")
        for page in paginator.paginate():
            for table_name in page["TableNames"]:
                tables.append(table_name)
        for table in tables:
            if table in ['log_dam_data', 'log_dev_cmd', 'log_dev_ctrl', 'dev_history_data', 'dac_use_stats']:
                continue
            if table in ['DAC21019XXXX_RAW','DAC20719XXXX_RAW']:
                result = cls.query_single_table(table, **kwargs[1])           
            else:
                result = cls.query_single_table(table, **kwargs[0])       
            if len(result) > 0:
                results = results.append(result)
                
        return results


def get_query_data(dev_id, start_time, end_time, attributes_to_get=None):

    query_parameters = [{
        "ConsistentRead":False,
        #"ProjectionExpression":'dac_id, #tmstp, telemetry',
        #"FilterExpression":'#tmstp BETWEEN :start_time AND :end_time',
        "KeyConditionExpression":'dev_id =:dev_id AND #tmstp BETWEEN :start_time AND :end_time',
        "ExpressionAttributeNames": {
            '#tmstp': 'timestamp',
        },
        "ExpressionAttributeValues": {
            ":dev_id":{"S": dev_id},
            ":start_time":{"S": start_time},
            ":end_time":{"S": end_time},
        },
    },
    {
        "ConsistentRead":False,
        #"ProjectionExpression":'dac_id, #tmstp, telemetry',
        #"FilterExpression":'#tmstp BETWEEN :start_time AND :end_time',
        "KeyConditionExpression":'dac_id =:dac_id AND #tmstp BETWEEN :start_time AND :end_time',
        "ExpressionAttributeNames": {
            '#tmstp': 'timestamp',
        },
        "ExpressionAttributeValues": {
            ":dac_id":{"S": dev_id},
            ":start_time":{"S": start_time},
            ":end_time":{"S": end_time},
        },
    }]

    if attributes_to_get is not None and isinstance(attributes_to_get, Iterable):
        attributes_to_get = list(x if x != "timestamp" else "#tmstp" for x in attributes_to_get) #timestamp is a reserved keyword
        query_parameters[0]["ProjectionExpression"] = ",".join(attributes_to_get)
        query_parameters[1]["ProjectionExpression"] = ",".join(attributes_to_get)

    return query_parameters


def __test():
    import datetime

    #end_time = (datetime.datetime.now() - datetime.timedelta(hours=2)).replace(minute=0, second=0, microsecond=0)
    end_time = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)
    start_time = (end_time - datetime.timedelta(hours=1)).isoformat()
    end_time = end_time.isoformat()
    
    # start_time = datetime.datetime(2020,12,20).isoformat()
    # end_time = datetime.datetime(2021,3,24).isoformat()

    r = ['DAC202200004']
    results = {}
    for dac_id in r:
        query_data = get_query_data(dac_id, start_time, end_time, attributes_to_get=None)
        results[dac_id] = dynamo_querier.query_all_proc(query_data)

    print(results)


if __name__ == "__main__":
    __test()