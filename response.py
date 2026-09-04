import requests as rq
import pandas as pd
df = pd.read_csv("data/processed/daily_latest.csv")
df = df.drop("Date",axis = 1)
parameters = df.tail(1)
payload = parameters.to_dict(orient='records')[0]
response = rq.post(url= "http://127.0.0.1:8000/Post_values", json=payload)
print(response.json())