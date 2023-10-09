#%%
import requests
import time
import pprint
from utils import * 
#%%
session = requests.Session()

set_url(parse_args())
#%%
while True:
    try:
        pprint.pprint(fetch_stats(session))
    except:
        pass
    time.sleep(1)

#%%
