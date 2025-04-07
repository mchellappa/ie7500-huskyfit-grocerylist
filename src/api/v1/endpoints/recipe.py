from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import spacy
import os
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Loading the trained models
rf_pipeline = joblib.load('./src/models/classification/randomforest_model.pkl')
gb_pipeline = joblib.load('./src/models/classification/gradientboosting_model.pkl')
svc_pipeline = joblib.load('./src/models/classification/linearsvc_model.pkl')
label_encoder = joblib.load('./src/models/classification/label_encoder.pkl')
router = APIRouter()


def predict_cuisine(ingredients, rf_pipeline, gb_pipeline, svc_pipeline, label_encoder):
    input_str = ' '.join(ingredients)
    rf_probs = rf_pipeline.predict_proba([input_str])[0]
    gb_probs = gb_pipeline.predict_proba([input_str])[0]
    svc_probs = svc_pipeline.predict_proba([input_str])[0]
    ensemble_probs = (rf_probs + gb_probs + svc_probs) / 3
    predicted_class = np.argmax(ensemble_probs)
    predicted_cuisine = label_encoder.inverse_transform([predicted_class])[0]
    confidence_score = ensemble_probs[predicted_class]
    return predicted_cuisine, confidence_score, ensemble_probs

# Load the SpaCy model
ner_model_path = "./src/models/ner_model"
if not os.path.exists(ner_model_path):
    raise IOError(f"Model not found at {ner_model_path}")

nlp = spacy.blank("en")

class RecipeRequest(BaseModel):
    recipe: str

class Entity(BaseModel):
    quantity: str  # Change to string to accommodate empty quantities
    ingredient: str

class RecipeResponse(BaseModel):
    cuisine: str
    entities: list[Entity]

@router.post("/process-recipe/")
async def process_recipe(request: RecipeRequest):
    recipe_text = request.recipe
    try:
        nlp_ner = spacy.load(ner_model_path)
    except IOError:
        raise HTTPException(status_code=500, detail=f"Model not found at {ner_model_path}")
    
    doc = nlp_ner(recipe_text)
    
    entities = []  # Initialize as an empty list
    ingredients_list = []
    print(doc.ents)
    for ent in doc.ents:
        if ent.label_ == "QUANTITY":
            for i_ent in doc.ents:
                if i_ent.label_ == "INGREDIENT" and i_ent.start == ent.end:
                    entity = Entity(ingredient=i_ent.text, quantity=ent.text)
                    entities.append(entity)                    
                    break
        elif ent.label_ == "INGREDIENT":
            # Create an array of ingredients excluding quantity
            ingredients_list.append(ent.text)
            
            if not any(e.ingredient == ent.text for e in entities):
                entity = Entity(ingredient=ent.text, quantity="")
                entities.append(entity)
    
    predicted_cuisine, _, _ = predict_cuisine(
        ingredients_list, rf_pipeline, gb_pipeline, svc_pipeline, label_encoder
    )
    return RecipeResponse(cuisine=predicted_cuisine, entities=entities)