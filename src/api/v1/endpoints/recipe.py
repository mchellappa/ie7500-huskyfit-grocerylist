from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import spacy
import os

router = APIRouter()

# Load the SpaCy model
model_path = "./src/ner_model"
if not os.path.exists(model_path):
    raise IOError(f"Model not found at {model_path}")

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
        nlp_ner = spacy.load(model_path)
    except IOError:
        raise HTTPException(status_code=500, detail=f"Model not found at {model_path}")
    
    doc = nlp_ner(recipe_text)
    
    entities = []  # Initialize as an empty list
    for ent in doc.ents:
        if ent.label_ == "QUANTITY":
            for i_ent in doc.ents:
                if i_ent.label_ == "INGREDIENT" and i_ent.start == ent.end:
                    entity = Entity(ingredient=i_ent.text, quantity=ent.text)
                    entities.append(entity)                    
                    break
        elif ent.label_ == "INGREDIENT":
            if not any(e.ingredient == ent.text for e in entities):
                entity = Entity(ingredient=ent.text, quantity="")
                entities.append(entity)

    return RecipeResponse(cuisine='indian', entities=entities)