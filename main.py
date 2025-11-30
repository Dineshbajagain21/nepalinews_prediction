from typing import Union
import pandas as pd
import pickle as pk
from fastapi import FastAPI

app = FastAPI()

dbfile = open('salary.pickle', 'rb')    
model = pk.load(dbfile)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/salary/")
def read_item(age: int,exp: int,gender: bool,b: bool,m: bool,p: bool):
    df = pd.DataFrame({
    'Age':[age],
    'Years of Experience':[exp],
    'Male':[gender],
    "Bachelor's":[b],
    "Master's":[m],
    "PhD":[p]
    })
    result = round(model.predict(df)[0][0],2)
    return {"result": result}