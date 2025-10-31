import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# Load the pipeline
with open("pipeline_v1.bin", "rb") as f:
    model = pickle.load(f)

class Client(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

app = FastAPI()

@app.post("/")
def predict(client: Client):
    X = client.dict()
    proba = model.predict_proba([X])[0, 1]
    return {"probability": round(proba, 3)}