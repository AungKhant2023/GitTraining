from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load the trained text classification model
svm_model = joblib.load(r"D:\PythonWorks\AllData\AungKhant\svm_pipeline.pkl")

app = FastAPI()

# Define request body structure
class TextInput(BaseModel):
    text: str

# Root Endpoint (Optional)
@app.get("/")
def read_root():
    return {"message": "Welcome to the Text Classification API"}

# Define text cleaning function (update as needed)
def text_cleaning_and_tokenize(text):
    return text  # For now, return text as-is

# Endpoint for text classification
@app.post("/classify")
def classify_text(input_data: TextInput):
    cleaned_text = text_cleaning_and_tokenize(input_data.text)
    prediction = svm_model.predict([cleaned_text])
    return {"prediction": prediction[0]}

