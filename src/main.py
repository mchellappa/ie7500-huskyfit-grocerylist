from fastapi import FastAPI
from src.api.v1.endpoints.recipe import router as recipe_router

app = FastAPI()

app.include_router(recipe_router, prefix="/api/v1")