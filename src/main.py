from fastapi import FastAPI
"""
Ingredient Intelligence API
This FastAPI application serves as the backend for the Ingredient Intelligence project. 
The project aims to provide API for retrieving ingredients and quantities from recipes. 
Also to predict the cusine type of a recipe.
Owner: A collaboration between Senthilkumaran Ramanathan, Divya Maheshkumar and Muthu Chellappa.
"""
from src.api.v1.endpoints.recipe import router as recipe_router
from fastapi.responses import RedirectResponse

app = FastAPI(
    openapi_version="3.1.0",
    title="Ingredient Intelligence",
    version="0.1.0",
    description="API for retrieving ingredients and quantities from a unstructured recipe text. Also to predict the cusine type of the recipe.",
    terms_of_service="",
    contact={
        "name": "IE7500-NLP-Team A",
        "url": "https://github.com/mchellappa/ie7500-huskyfit-grocerylist",       

        }
)

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

app.include_router(recipe_router, prefix="/api/v1")