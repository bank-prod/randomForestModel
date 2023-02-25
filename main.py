from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle


app = FastAPI()

fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]

class Feature(BaseModel):
    CreditScore: float
    Gender : str
    Age :int
    Tenure: int
    Balance:float
    Geography:str
    NumOfProducts : int
    HasCrCard : int
    IsActiveMember:int
    EstimatedSalary:float


@app.get("/")
async def root():
    return {"message": "Hello World"}

# @app.get("/items/{item_id}")
# async def read_item(item_id):
#     return {"message": f"Id is {item_id}"}


@app.post("/feature/")
async def create_items(feature: Feature):
    feature_dict = dict(feature)

    model = pickle.load(open('mymodel.sav', 'rb'))

    classe , proba = model.predict(feature_dict)

    return {'Classe':classe,'Probability':proba}
