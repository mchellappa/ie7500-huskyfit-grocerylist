import re
import ast


def extract_quantities(text):
    """Extract quantities from the text."""
    quantity_patterns = [
        r'\b\d+\s*(?:cup|cups|tbsp|tablespoon|tablespoons|tsp|teaspoon|teaspoons|oz|ounce|ounces|lb|pound|pounds|g|gram|grams|kg|kilogram|kilograms|ml|milliliter|milliliters|l|liter|liters)\b',
        r'\b\d+/\d+\s*(?:cup|cups|tbsp|tablespoon|tablespoons|tsp|teaspoon|teaspoons|oz|ounce|ounces|lb|pound|pounds|g|gram|grams|kg|kilogram|kilograms|ml|milliliter|milliliters|l|liter|liters)\b',
        r'\b\d+\s*(?:and\s*\d+/\d+)?\s*(?:cup|cups|tbsp|tablespoon|tablespoons|tsp|teaspoon|teaspoons|oz|ounce|ounces|lb|pound|pounds|g|gram|grams|kg|kilogram|kilograms|ml|milliliter|milliliters|l|liter|liters)\b',
        r'\b\d+\s*(?:-\s*\d+)?\s*(?:cup|cups|tbsp|tablespoon|tablespoons|tsp|teaspoon|teaspoons|oz|ounce|ounces|lb|pound|pounds|g|gram|grams|kg|kilogram|kilograms|ml|milliliter|milliliters|l|liter|liters)\b',
        r'\b\d+\s*(?:to\s*\d+)?\s*(?:cup|cups|tbsp|tablespoon|tablespoons|tsp|teaspoon|teaspoons|oz|ounce|ounces|lb|pound|pounds|g|gram|grams|kg|kilogram|kilograms|ml|milliliter|milliliters|l|liter|liters)\b',
        r'\b\d+\s*(?:\.\d+)?\s*(?:cup|cups|tbsp|tablespoon|tablespoons|tsp|teaspoon|teaspoons|oz|ounce|ounces|lb|pound|pounds|g|gram|grams|kg|kilogram|kilograms|ml|milliliter|milliliters|l|liter|liters)\b',
        r'\b½\s*(?:cup|cups|tbsp|tablespoon|tablespoons|tsp|teaspoon|teaspoons|oz|ounce|ounces|lb|pound|pounds|g|gram|grams|kg|kilogram|kilograms|ml|milliliter|milliliters|l|liter|liters)\b'
    ]
    entities = []
    for pattern in quantity_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            entities.append((match.start(), match.end(), "QUANTITY"))
    return entities

def get_clean_ingredients(ingredient_list):
    """Clean the ingredients list to be used for NER."""
    try:
        return ast.literal_eval(ingredient_list)  # Convert string to list
    except (ValueError, SyntaxError):
        return []

def extract_ingredients(text, ingredients):
    """Extract ingredients from the text."""
    entities = []
    for ingredient in ingredients:
        start_idx = text.lower().find(ingredient.lower())
        if start_idx != -1:
            end_idx = start_idx + len(ingredient)
            entities.append((start_idx, end_idx, "INGREDIENT"))
    return entities