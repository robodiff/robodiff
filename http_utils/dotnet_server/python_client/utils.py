import time
import json
import numpy as np
import random
import argparse

headers = {
    'Content-type':'application/json', 
    'Accept':'application/json'
}
url = "http://localhost:24479"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="10.243.16.52")
    parser.add_argument("--port", default="24478")
    return parser.parse_args()

def set_url(args):
    global url
    url = f"http://{args.ip}:{args.port}"


# URL for DM laptop.
# url = "http://192.168.3.165:24478"
#url = 'http://localhost:24479' # Set destination URL here

def infinite():
    while True:
        yield

def validate_result(r):
    assert r.status_code == 200
    return json.loads(r.text)

def fetch_stats(session):
    r = session.get(f"{url}/stats/")
    return validate_result(r)
    
def register(session):
    r = session.post(f"{url}/register", headers=headers)
    return validate_result(r)

def fetch_sim(session, node_guid):
    r = session.get(f"{url}/simulation/{node_guid}", headers=headers)
    return validate_result(r)

def send_results(session, work_guid, results=None):
    r = session.post(f"{url}/results/{work_guid}", json=results)
    return validate_result(r)

def send_finished(session, work_guid):
    r = session.post(f"{url}/finished/{work_guid}")
    return validate_result(r)

def send_error(session, work_guid, details=None):
    r = session.post(f"{url}/errors/{work_guid}", json=details)
    return validate_result(r)
    
def process_simulation(session, node_guid, error_rate=0.01, silent_error_rate=0.01, retry_delay=None):
    raw_sim = fetch_sim(session, node_guid)
    work_guid = raw_sim['workGuid']
    if retry_delay is None:
        retry_delay = raw_sim['retryDelay']
    sim = raw_sim["innerSimulation"]
    
    if sim is None:
        time.sleep(retry_delay)
        return
    
    epochs = sim["epochs"]
    for epoch_idx in range(epochs):
        # did an 'error' occur?
        if (random.random() < error_rate):
            if (random.random() > silent_error_rate):
                send_error(session, work_guid)
            return # error occurred. 

        send_results(session,
            work_guid,
            {
                'epoch_idx': epoch_idx,
                'loss': random.random(),
                'morphModel':{
                    'genome':{
                        'z': np.random.random(np.array(sim['morphModel']['genome']['z']).shape).tolist()
                    }
                },
                'actuationModel':{
                    'genome':{
                        'z': np.random.random(np.array(sim['actuationModel']['genome']['z']).shape).tolist()                
                    }
                }
            })

    send_finished(session, work_guid)

