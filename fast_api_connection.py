from fastapi import FastAPI
api = FastAPI()
from pydantic import BaseModel,create_model
import pandas as pd
import joblib

pipline = joblib.load("pipeline_rasso.pkl")
model = pipline["model"]
features = pipline["features"]

d = {}
print(features)
for i,j in features.items():
    j = str(j)
    if j in ('int32','int64'):
        d[i] = (int,...)
    elif j in ('float32','float64'):
        d[i] = (float,...)
    elif j in ('object',):
        d[i] = (object,...)

User = create_model("User",**d)

@api.post("/Post_values")
def predictions(user:User):
    data = user.model_dump()
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {
        "predictions":prediction.tolist()
    }

