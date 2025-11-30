import pandas as pd
from fastapi import FastAPI
import pickle as pk
app = FastAPI()

dbfile = open('Covid_Classification.pickle', 'rb')    
model = pk.load(dbfile)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/covid")
def read_item(Cough_symptoms: bool, Fever:bool,Sore_throat:bool,Shortness_of_breath:bool,Headache:bool,Known_contact:int):
    df = pd.DataFrame({
        'Cough_symptoms':[Cough_symptoms],
        'Fever':[Fever],
        'Sore_throat':[Sore_throat],
        'Shortness_of_breath':[Shortness_of_breath],
        'Headache':[Headache],
        'Known_contact':[Known_contact]
        })
    result = model.predict(df)[0]
    if result == 1:
        return {"result":"Covid Positive"}
    else:
        return {"result":"Covid Negitive"}