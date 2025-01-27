import re
import ast
# def extract_quantities(text):
#     """Extract quantities and their units from the text."""
#     #quantity_pattern = r"(\d+\s?(?:c|tbsp|tsp|oz|lbs?|g|ml|liters?|cups?|cup|pkg|pound|pounds|grams?|teaspoons?|tablespoons?|\d+/\d+)\.?)"
#     quantity_pattern = r"(\d+/\d+|\d+(\.\d+)?\s?(cups?|tablespoons?|teaspoons?|oz|grams?|kg|ml|liters?|tbsp|tsp|lb|lbs?|pounds?|g))"

#     quantities = []
#     for match in re.finditer(quantity_pattern, text):
#         quantities.append((match.start(), match.end(), "QUANTITY"))
#     return quantities

quantity_pattern = r"(\d+/\d+|\d+(\.\d+)?\s?(cups?|tablespoons?|teaspoons?|oz|grams?|kg|ml|liters?|tbsp|tsp|lb|lbs?|pounds?|g))"
def extract_quantities(text):
    """Extract quantity and unit from the text."""
    quantities = []
    for match in re.finditer(quantity_pattern, text):
        quantities.append((match.start(), match.end(), "QUANTITY"))
    return quantities

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
