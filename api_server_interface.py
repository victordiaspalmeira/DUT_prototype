import dotenv
dotenv.load_dotenv("./.env")

import requests
from typing import Any, Dict, Iterable, Union
import os
import datetime
from pprint import pprint
import functools as ft
#from intel_info_connect import log_api_server_request

def __request(path, headers=None, json={}):
    r = requests.post(path, json=json, headers=headers)
    # log_api_server_request({
    #     "path" : path,
    #     "status_code" : r.status_code,
    #     "timestamp" : datetime.datetime.now().isoformat()
    # })
    return r

class API_Connection:
    @staticmethod
    def __request(path, headers=None, json={}):
        r = requests.post(path, json=json, headers=headers)
        # try:
        #     log_api_server_request({
        #         "uri" : path,
        #         "status_code" : r.status_code,
        #         "timestamp" : datetime.datetime.now().isoformat()
        #     })
        # except Exception as e:
        #     print("Logging failed! Exception ", e)
        return r
    def __init__(self, base_url=os.getenv("API_BASEURL")) -> None:
        self.base_url = base_url
        self.intel_token = os.getenv("API_TOKEN")
        self.jwt_token = self.intel_token

    def login(self, username, password) -> requests.Response:
        r = API_Connection.__request(f"{self.base_url}/login", json={"user":username, "password":password})
        if r.status_code == 200:
            self.jwt_token = r.json()["token"]
        return r
    
    def get_dac_list(self, **kwargs):
        r = API_Connection.__request(f"{self.base_url}/dac/get-dacs-list", json=kwargs, headers={"Authorization": f"Bearer {self.intel_token}"})
        return r.json()

    def get_dac_data(self, dac_id) -> Dict[str, Any]:
        r = API_Connection.__request(f"{self.base_url}/dac/get-dac-info", json={"DAC_ID":dac_id}, headers={"Authorization": f"Bearer {self.jwt_token}"})
        return r.json()['info'] #will throw if request failed
    
    def get_health_hist(self, dac_id, start_date : Union[str, datetime.datetime]=None):
        if start_date:
            since_date = start_date.isoformat() if isinstance(start_date, datetime.datetime) else start_date 
            r = API_Connection.__request(f"{self.base_url}/dac/get-health-hist", json={"dacId":dac_id, "SINCE":since_date}, headers={"Authorization": f"Bearer {self.jwt_token}"})
        else:
            r = API_Connection.__request(f"{self.base_url}/dac/get-health-hist", json={"dacId":dac_id}, headers={"Authorization": f"Bearer {self.jwt_token}"})
        return r.json()['list']
    
    def get_frep_hist(self, dac_id):
        r = API_Connection.__request(f"{self.base_url}/dac/get-frep-hist", json={"dev_id" : dac_id}, headers={"Authorization": f"Bearer {self.intel_token}"})
        return r.json()
    
    def send_fault_detected(self, fault : Dict):
        for f in fault['warnings']:
            for key in f.keys():
                if isinstance(f[key], datetime.datetime):
                    f[key] = f[key].isoformat()

        fault['timestamp'] = datetime.datetime.now().isoformat(timespec='seconds')
        print("FAULT: ", fault)
        r = API_Connection.__request(f"{self.base_url}/dac/fault-detected", json=fault, headers={"Authorization": f"Bearer {self.intel_token}"})
        if r.status_code == 200:
            print(f"fault {fault['ID']} sent!")
        return r.status_code

    def get_dac_day_charts_data(self, dev_id, day, selected_params=["Tamb", "Tliq", "Tsuc", "Pliq", "Psuc", "L1"], withFaults=True):
        """Params:
            dacId: string
            day: string
            selectedParams: string[]
            withFaults?: boolean
        """
        payload = {
            "dacId" : dev_id,
            "day" : day,
            "selectedParams" : selected_params
        }

        r = API_Connection.__request(f"{self.base_url}/dac/get-day-charts-data", headers={"Authorization": f"Bearer {self.intel_token}"}, json=payload)
        return r.json()

    def get_dac_recent_history(self, dev_id, interval_length_s):
        payload = {
            "dacId" : dev_id,
            "intervalLength_s" : interval_length_s
        }
        r = API_Connection.__request(f"{self.base_url}/dac/get-recent-history-v2", headers={"Authorization": f"Bearer {self.intel_token}"}, json=payload)
        return r.json()

    def get_duts_list(self, **kwargs):
        """Params:
            clientId?: int,
            clientIds?: List[int],
            stateId?: str,
            cityId?: str,
            unitId?: int,
            groupId?: int,
            rtypeId?: int,
            SKIP?: int,
            LIMIT?: int,
            onlyWithAutomation?: bool
        """
        r = API_Connection.__request(f"{self.base_url}/dut/get-duts-list", headers={"Authorization": f"Bearer {self.intel_token}"}, json=kwargs)

        return r.json()
    
    def get_dut_info(self, dev_id):
        r = API_Connection.__request(f"{self.base_url}/dut/get-dut-info", headers={"Authorization": f"Bearer {self.intel_token}"}, json={"DEV_ID" : dev_id})
        return r.json()

    def get_groups_list(self, **kwargs):
        """Params:
        UNIT_ID?: number,
        CLIENT_ID?: number,
        CLIENT_IDS?: number[],
        includeDacsCount?: boolean
        """
        r = API_Connection.__request(f"{self.base_url}/clients/get-groups-list", headers={"Authorization": f"Bearer {self.intel_token}"}, json=kwargs)
        return r.json()

    def get_associated_dacs(self, dev_id):
        groups_dut = list(filter(lambda x : x['DUT_ID'] == dev_id, self.get_groups_list()))
        if len(groups_dut) > 0:
            return self.get_dac_list(groupId=groups_dut[0]['GROUP_ID'])
        return {"list" : []}

    def get_dut_day_charts_data(self, dev_id, day):
        payload = {
            "devId" : dev_id,
            "day" : day,
            "selectedParams" : ["Temperature", "Humidity"]
        }
        r = API_Connection.__request(f"{self.base_url}/dut/get-day-charts-data", headers={"Authorization": f"Bearer {self.intel_token}"}, json=payload)
        print(r.text)
        return r.json()

api_conn = API_Connection(os.getenv("API_BASEURL"))

if __name__ == "__main__":
    a = API_Connection()
    #pprint(a.get_duts_list())
    #pprint(a.get_dut_info("DUT209201107"))
    #pprint(a.get_groups_list(UNIT_ID=120))
    #pprint(a.get_associated_dac('DUT209201107'))

    dut_list = a.get_duts_list()['list']
    #print(dut_list)
    for dut in dut_list:
        associated_dacs = a.get_associated_dacs(dut['DEV_ID'])
        if len(associated_dacs) > 0:
            print(dut, associated_dacs['list'])
            print('--------------------------------')


    #print(sum(day["Temperature"]["c"]))

    #print(a.get_dut_day_charts_data("DUT209201107", "2021-04-22"))
