#%%
import requests
import tqdm
from utils import * 
#%%
while True:
    try:
        session = requests.Session()
        node_guid = register(session)

        for _ in tqdm.tqdm(infinite(), ascii=True):
            process_simulation(session, node_guid, error_rate=0.001, silent_error_rate=0.1, retry_delay=10)
    except Exception as e:
        time.sleep(10)
# %%
