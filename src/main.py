from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy
import os

app = FastAPI()

# Load the SpaCy model
model_path = "./src/output/model-best"
if not os.path.exists(model_path):
    raise IOError(f"Model not found at {model_path}")

nlp = spacy.blank("en")

class RecipeRequest(BaseModel):
    ingredients: str
class Entity(BaseModel):
    text: str
    start: int
    end: int
    label: str

class RecipeResponse(BaseModel):
    entities: list[Entity]


@app.post("/process-recipe/")
async def process_recipe(request: RecipeRequest):
    recipe_text = request.ingredients
    try:
        nlp_ner = spacy.load(model_path)
    except IOError:
        raise HTTPException(status_code=500, detail=f"Model not found at {model_path}")
    
    doc = nlp_ner(recipe_text)
    entities = [
        {"text": ent.text, "start": ent.start_char, "end": ent.end_char, "label": ent.label_}
        for ent in doc.ents
    ]
    return RecipeResponse(entities= entities)
    
