import pandas as pd
import spacy
from spacy.tokens import DocBin
import re
import ast

from helper import extract_ingredients, extract_quantities, get_clean_ingredients

# Load the dataset
df = pd.read_csv("recipe_dataset_large.csv")  # Replace with your actual file path

# Initialize SpaCy blank English model
nlp = spacy.blank("en")

# Define chunk size
chunk_size = 1000  # Adjust based on your memory capacity

# Process the dataset in chunks
for i in range(0, len(df), chunk_size):
    doc_bin = DocBin()
    chunk = df.iloc[i:i + chunk_size]
    
    for _, row in chunk.iterrows():
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

    # Save the processed training data in SpaCy format for each chunk
    doc_bin.to_disk(f"ner/ner_training_data_chunk_{i // chunk_size}.spacy")

print("SpaCy training data saved in chunks.")