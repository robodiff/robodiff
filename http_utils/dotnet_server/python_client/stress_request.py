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
            fetch_sim(session, node_guid)
    except:
        time.sleep(1)
        pass


# %%
