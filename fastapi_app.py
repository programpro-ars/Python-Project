from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd

app = FastAPI()
df = pd.read_csv('cardio.csv', sep=';')

class NewEntry(BaseModel):
    age: int
    height: int
    weight: float
    gender: int
    ap_hi: int
    ap_lo: int
    cholesterol: int
    gluc: int
    smoke: int
    alco: int
    active: int
    cardio: int

@app.get("/data/")
def get_data(start: int = 0, limit: int = 10, filter_cardio: int = Query(None)):
    """
    GET method with pagination and filtering by 'cardio' status.
    """
    filtered_df = df if filter_cardio is None else df[df['cardio'] == filter_cardio]
    return filtered_df.iloc[start:start + limit].to_dict(orient="records")

@app.post("/data/")
def add_entry(new_entry: NewEntry):
    """
    POST method to add a new instance to the dataset.
    """
    global df
    # Add the new entry to the DataFrame
    new_row = new_entry.model_dump()
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return {"message": "New entry added successfully!", "data": new_row}
