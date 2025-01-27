import pandas as pd
import spacy
from spacy.tokens import DocBin
import re
import ast

# Load the dataset
df = pd.read_csv("recipe_dataset.csv")  # Replace with your actual file path

# Initialize SpaCy blank English model
nlp = spacy.blank("en")
doc_bin = DocBin()

# Regular expression for detecting quantities and units (cup, tsp, etc.), including fractions
quantity_pattern = r"(\d+/\d+\s?(cups?|tablespoons?|teaspoons?|oz|grams?|kg|ml|liters?|tbsp|tsp|lb|lbs?|pounds?|g|c\.))|(\d+(\.\d+)?\s?(cups?|tablespoons?|teaspoons?|oz|grams?|kg|ml|liters?|tbsp|tsp|lb|lbs?|pounds?|g|c\.))"

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

# Loop through the dataset to process each recipe
for _, row in df.iterrows():
    recipe_text = row['ingredients']  # Assuming 'ingredients' column has the list of ingredients
    ner_labels = get_clean_ingredients(row['NER'])  # Clean the NER column

    # Extract both quantities and ingredients from the text
    quantity_entities = extract_quantities(recipe_text)
    ingredient_entities = extract_ingredients(recipe_text, ner_labels)

    # Combine quantity and ingredient entities
    all_entities = quantity_entities + ingredient_entities

    # Make sure there are no overlapping entities
    added_spans = []
    entities_to_add = []
    doc = nlp.make_doc(recipe_text)

    for start, end, label in all_entities:
        # Check for overlapping spans
        if all(not (start < existing_end and end > existing_start) for existing_start, existing_end in added_spans):
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span:
                entities_to_add.append(span)
                added_spans.append((start, end))

    # Assign entities to the doc and add to DocBin
    if entities_to_add:
        doc.ents = entities_to_add
        doc_bin.add(doc)

# Save the processed training data in SpaCy format
doc_bin.to_disk("ner_training_data.spacy")

print("SpaCy training data saved as 'ner_training_data.spacy'.")
