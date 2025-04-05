import pandas as pd
import sys
import numpy as np
from unidecode import unidecode
import re
from fractions import Fraction
import spacy
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch.amp import autocast

def combine_relevant_features(df_cleaned):
    """
    Function to combine the relevant features - title, ingredients, directions for further processing
    Parameters:
        df_cleaned: The DataFrame containing the relevant columns.
    Returns:
        df_cleaned: The cleaned DataFrame with the additional combined feature column.
    """
    # Convert ingredients, directions, and NER to lowercase lists
    df_cleaned['ingredients'] = df_cleaned['ingredients'].apply(lambda x: [i.lower() for i in ast.literal_eval(x)])
    df_cleaned['directions'] = df_cleaned['directions'].apply(lambda x: [i.lower() for i in ast.literal_eval(x)])
    
    # Concatenate relevant columns for NER
    df_cleaned['combined'] = df_cleaned['title'] + ' ' + df_cleaned['ingredients'].astype(str) + ' ' + df_cleaned['directions'].astype(str)

    # Lowercase the 'combined' column
    df_cleaned['combined'] = df_cleaned['combined'].apply(lambda x: x.lower())

    return df_cleaned


def normalize_to_ascii(text):
    return unidecode(text)


def clean_labels(df_cleaned):
    """
    Function to clean the recipe dataset by removing NER labels with empty string and unwanted symbols.
    Parameters:
        df_cleaned: The DataFrame containing the recipe data with NER label column.
    Returns:
        df_cleaned: The cleaned DataFrame with rows removed from cleaning the NER label column
    """
    # Convert all NER labels to lowercase and evaluate them as lists
    df_cleaned['NER'] = df_cleaned['NER'].apply(lambda x: [i.lower() for i in ast.literal_eval(x)])
    
    # Clean each item in the 'NER' column to remove unwanted symbols but preserve alphanumeric characters, hyphens, and spaces
    def clean_item(item):
        item = normalize_to_ascii(item)
        item = ''.join(e for e in item if e.isalnum() or e in ['-', ' '])  # Remove unwanted symbols but keep alphanumeric and spaces
        if any(char.isdigit() for char in item):
            return ""  # Return empty string, depending on how you want to handle it
        return item
    
    # Apply the clean_item function to each item in the 'NER' list
    df_cleaned['NER'] = df_cleaned['NER'].apply(lambda x: [clean_item(item) for item in x])

    df_cleaned['NER'] = df_cleaned['NER'].apply(lambda x: [item for item in x if item != ""])
    
    df_cleaned = df_cleaned[df_cleaned['NER'].apply(lambda x: x != [])].reset_index(drop=True)
    
    return df_cleaned

def remove_rows_with_inaccurate_labels(df_cleaned, threshold=90):
    """
    Function to clean the recipe dataset by removing rows where less than 90% percent of NER labels exist in the data.
    Parameters:
        df_cleaned: The DataFrame containing the recipe data with 'ingredients' and 'NER' columns.
        threshold (int): The minimum percentage of correct labels for ingredients in NER to keep the row (default is 90%).
    Returns:
        df_cleaned: The cleaned DataFrame with rows removed where less than the 90% percent of ingredients are correctly identified.
    """

    # Function to check if any NER label is found in the ingredient phrase
    def verify_ner_labels(ingredients, ner_labels):
        incorrect_ingredients = []
        for ingredient in ingredients:
            match_found = False
            for label in ner_labels:
                # Check if the label is a substring of the ingredient phrase (case-insensitive)
                if label.lower() in ingredient.lower():
                    match_found = True
                    break
            if not match_found:
                incorrect_ingredients.append(ingredient)
        return incorrect_ingredients
    
    # Function to check if the percentage of correct matches is above the threshold
    def is_partial_match(incorrect_ingredients, ingredients, threshold=90):
        incorrect_count = len(incorrect_ingredients)
        total_count = len(ingredients)
        match_percentage = (total_count - incorrect_count) / total_count * 100
        return match_percentage < threshold

    # Create a new DataFrame to avoid modifying the original one
    df_filter = df_cleaned[['ingredients', 'NER']]

    # Apply the NER verification function to each row in the dataframe
    df_filter.loc[:, 'incorrect_ingredients'] = df_filter.apply(
    lambda row: verify_ner_labels(row['ingredients'], row['NER']), axis=1
    )

    # Filter rows where less than 90% of ingredients are matched in the NER labels
    partial_ner_df = df_filter[df_filter.apply(
        lambda row: is_partial_match(row['incorrect_ingredients'], row['ingredients'], threshold=threshold), axis=1
    )]

    # Check if any rows have partial NER matches and drop them from df_cleaned
    if len(partial_ner_df) > 0:
        print(f"Dropped {len(partial_ner_df)} rows with less than {threshold}% NER matches")
        # Get the indices of rows with partial matches
        partial_ner_indices = partial_ner_df.index
        
        # Drop these rows from the original df_cleaned
        df_cleaned = df_cleaned.drop(partial_ner_indices)
    else:
        print(f"No rows with less than {threshold}% NER matches")

    # Return the cleaned DataFrame
    print(f"Cleaned dataframe shape: {df_cleaned.shape}")
    return df_cleaned

def remove_unwanted_characters(df_cleaned):
    """
    Standardize the text in the specified column of the dataframe by applying various regex transformations.
    Parameters:
    df_cleaned: The input dataframe.
    Returns:
    df_cleaned: The dataframe with the 'normalized_combined' column added.
    """
    
    # Apply the transformations to the specified column
    df_cleaned['normalized_combined'] = df_cleaned['combined'].apply(
        lambda x: re.sub(r'\\+[a-zA-Z]|[^a-z0-9\s./-]', ' ',  # Keep alphanumeric characters, spaces, periods, slashes, and hyphens
                        # Replace multiple spaces with a single space
                        re.sub(r'(?<!\d)\.(?!\d)', ' ',  # Remove periods that are not part of numbers
                               re.sub(r'(?<=\d|[^\w])-', ' ',  # Remove hyphens that come after numbers
                                      re.sub(r'(\d)\s*/\s*(\d)', r'\1/\2',  # Normalize fractions by removing spaces around the slash (e.g., "1 / 2" -> "1/2")
                                             re.sub(r'(?<!\d)\s*/\s*(?!\d)', ' ',  # Remove slashes that are not followed by a number (e.g., '/brown' -> 'brown')
                                                    str(x)  # The input data to apply the normalization
                                                   ))))).strip()  # Strip leading/trailing spaces
    )
    
    # Remove extra spaces
    df_cleaned['normalized_combined'] = df_cleaned['normalized_combined'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    return df_cleaned

def rectify_the_l_character(df_cleaned):
    '''
    Rectify values where the character l was incorrectly used in the data
    Parameters:
    df_cleaned: The input dataframe.
    Returns:
    df_cleaned: The dataframe with the 'normalized_combined' column added.
    '''
#Find rows where l is part of fraction
    filtered_rows = df_cleaned[df_cleaned['normalized_combined'].str.contains(r'\bl/\b', case=False, na=False)]
#Find index of rows with l from the filtered rows
    matching_rows = filtered_rows[filtered_rows['normalized_combined'].str.contains('l', regex=False)].index
#Replace all l in filtered rows with 1 as these rows have incorrectly represented 1 as l
    df_cleaned.loc[matching_rows, 'normalized_combined'] = df_cleaned.loc[matching_rows, 'normalized_combined'].str.replace('l', '1', regex=False)

# Replace 'l' with '1' only when it is followed by ' lb' in the entire dataset
    df_cleaned['normalized_combined'] = df_cleaned['normalized_combined'].str.replace(r'(\d*)\s*l\s+lb', r'\1 1 lb', regex=True)

# Replace 'l' with 'liter' for cases where a number is followed by the letter 'l'
    df_cleaned['normalized_combined'] = df_cleaned['normalized_combined'].str.replace(r'\b\d+(\.\d+)?\s+l(?!/)\b', lambda x: x.group(0).replace(' l', ' liter'), regex=True)

# Add space where an alphabet or word is followed by / so that it is not interpreted as a fraction. E.g,25 g/1/4 cup becomes 25 g / 1/4 cup
    df_cleaned['normalized_combined'] = df_cleaned['normalized_combined'].apply(lambda x: re.sub(r'(?<=\w)(?<!\d)/(?=\S)', r' / ', x))

    return df_cleaned

def convert_fraction_to_decimal(quantity):
    '''
       Function to convert mixed fractions (e.g., 1 1/2) or simple fractions to decimals (e.g., 1.5)
       Parameters:
       quantity: the value of the quanity in fractional form that needs to be converted to decimals
       Returns:
       quantity: fractional quantity converted to decimal values
    '''
    parts = quantity.split()  # Split by space to handle mixed fractions
    if len(parts) == 2:  # Handle mixed fraction
        whole_number = int(parts[0])
        fraction = parts[1]
        try:
            return round(whole_number + float(Fraction(fraction)), 2)
        except ZeroDivisionError:
            return quantity
    elif len(parts) == 1:  # Handle simple fractions or whole numbers
        if '/' in parts[0]:  # Simple fraction
            try:
                result = round(float(Fraction(parts[0])), 2)  # Convert fraction to decimal
                return result
            except ZeroDivisionError:
                return parts[0]
       
    return 0.0  # Fallback for any unexpected case

def normalize_text(df_cleaned):
    '''
       Function to normalize the units of measurements, quantities
       Parameters:
       df_cleaned: the dataframe that contains the cleaned column for normalizing the units of measurements
       Returns:
       df_cleaned: returns a normalized version of the 'normalized_combined' column
    '''

    df_cleaned = rectify_the_l_character(df_cleaned)
    
    #Custom normalization mapping
    normalization_map = {
        'tsp': 'teaspoon',
        'teaspoons': 'teaspoon',
        'tbsp': 'tablespoon',
        'tablespoons': 'tablespoon',
        'oz': 'ounce',
        'ounces': 'ounce',
        'c': 'cup',
        'cup': 'cup',
        'cups': 'cup',
        'lb': 'pound',
        'lbs': 'pound',
        'pound': 'pound',
        'pounds': 'pound',
        'g': 'gram',
        'grams': 'gram',
        'kg': 'kilogram',
        'kilograms': 'kilogram',
        'pkg': 'package',
        'qt': 'quart',
        'quarts': 'quart',
        'l': 'liter',
        'liters': 'liter',
        'ml': 'milliliter'
    }
    
    # Regular expression pattern to match simple, mixed fractional numbers and words
    pattern = r'(\d+/\d+|\d+\s*\d*/\d+|\d*\.\d+|\d+|\S+)'

    def process_text(text):
        # Find all matches in the text
        matches = re.findall(pattern, text)

        normalized_words = []
        for match in matches:
     
            # Check if the match is a fraction or a number
            if '/' in match:
                if re.match(r'^\d+\s*\d*/\d+$', match):  # Valid fraction like 1/2, 3/4, etc.
                    if re.match(r'^\d+\s+/\d+$|^/$', match): #Invalid fraction due to spaces like 1 /2n then skip
                        pass
                    else:
                        match = str(convert_fraction_to_decimal(match))
        
            # Normalize the unit using the normalization map if it's a word
            normalized_word = normalization_map.get(match.lower(), match)
            normalized_words.append(normalized_word)
    
        # Reconstruct the text from the normalized words
        return ' '.join(normalized_words)
    # Apply the process_text function to the column
    df_cleaned['normalized_combined'] = df_cleaned['normalized_combined'].apply(lambda x: process_text(str(x)))
    return df_cleaned

def load_known_ingredients(csv_path):
    # Load the CSV file into a DataFrame
    known_ingredients_df = pd.read_csv(csv_path)
    # Convert the 'ingredient' column into a set for faster lookups
    known_ingredients = set(known_ingredients_df['ingredients'].str.lower())
    
    return known_ingredients

def extract_ingredient_nouns(df_cleaned, known_ingredients):
    """
    Extracts valid ingredient nouns or phrases from the NER (Named Entity Recognition) tags
    and identifies non-ingredient terms.

    Args:
        df_cleaned (pandas.DataFrame): dataframe containing NER tags 
        known_ingredients (set): A set of known ingredient names, used to match exact terms.

    Returns:
        tuple: A tuple containing:
            - valid_phrases (list): A list of ingredient-related nouns or phrases (valid ingredients).
            - non_ingredients (list): A list of terms that are not identified as valid ingredients.
    """  
    # Load the English language model from spaCy
    nlp = spacy.load("en_core_web_sm")

    # Clean the NER tags
    ner_tags_cleaned = [[token for token in tags if len(token) > 1] for tags in df_cleaned['NER']]
    flattened_ner_set = {item for sublist in ner_tags_cleaned for item in sublist}

    # Word list with single word and multi-word NER phrases
    word_list = flattened_ner_set

    # Initialize a list to store valid nouns
    valid_phrases = []

    # Iterate through each word/phrase in the list
    for phrase in word_list:
        # Process each word or multi-word phrase separately
        doc = nlp(phrase)

        # Check if the token is a noun or proper noun or if it's in the known ingredients list
        if any(token.pos_ in ['NOUN', 'PROPN'] for token in doc) or phrase.lower() in known_ingredients:
            valid_phrases.append(phrase)

    # Identify non-ingredients
    non_ingredients = [word for word in word_list if word not in valid_phrases]

    return valid_phrases, non_ingredients


def load_non_ingredients(csv_path):
    # Load the CSV file into a DataFrame
    non_ingredients_df = pd.read_csv(csv_path)
    # Convert the 'ingredient' column into a set for faster lookups
    non_ingredients = non_ingredients_df['non_ingredients'].str.lower().tolist()
    
    return non_ingredients

# Function to identify non-ingredients using DistilBERT model
def identify_non_ingredients(tokens, non_ingredients_list, batch_size=8, similarity_threshold=0.972):
    """
    Identifies non-ingredient tokens by calculating their cosine similarity with known non-ingredient words.

    Args:
    - tokens (list): List of tokens to process (e.g., valid nouns).
    - non_ingredients_list (list): List of known non-ingredient words/phrases.
    - batch_size (int): Number of tokens to process in each batch.
    - similarity_threshold (float): Threshold for cosine similarity to consider a token as a non-ingredient.

    Returns:
    - identified_non_ingredients (list): List of tokens identified as non-ingredients.
    """
    # Check if CUDA (GPU) is available and move the model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pre-trained tokenizer and model (DistilBERT)
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)

    # Initialize an empty list to store identified non-ingredients
    identified_non_ingredients = []

    # Function to process tokens in smaller batches
    def process_batch(tokens, batch_size, non_ingredients_list):
        # Get embeddings for non-ingredient words
        non_ingredient_inputs = tokenizer(non_ingredients_list, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            with autocast('cuda'):
                non_ingredient_outputs = model(**non_ingredient_inputs)
        non_ingredient_embeddings = non_ingredient_outputs.last_hidden_state.mean(dim=1)

        for start_idx in range(0, len(tokens), batch_size):
            end_idx = min(start_idx + batch_size, len(tokens))
            token_batch = tokens[start_idx:end_idx]

            # Encode tokens using DistilBERT
            inputs = tokenizer(token_batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

            # Get DistilBERT embeddings for the input tokens
            with torch.no_grad():
                with autocast('cuda'):
                    outputs = model(**inputs)
            token_embeddings = outputs.last_hidden_state

            # Calculate cosine similarity between token embeddings and non-ingredient embeddings
            for i, token in enumerate(token_batch):
                # Get the embedding for the specific token
                token_embedding = token_embeddings[i][1:-1].mean(dim=0)
                token_embedding = token_embedding.detach().cpu().numpy()

                # Calculate cosine similarities
                similarities = cosine_similarity(token_embedding.reshape(1, -1), non_ingredient_embeddings.detach().cpu().numpy())
                max_similarity = similarities.max()

                # Identify non-ingredients based on direct match or similarity
                if token in non_ingredients_list or max_similarity > similarity_threshold:
                    identified_non_ingredients.append((token, max_similarity))
                    #print(f"{token} is likely not an ingredient")

            torch.cuda.empty_cache()  # Clear GPU memory after each batch

    # Process tokens in batches
    process_batch(tokens, batch_size, non_ingredients_list)

    return identified_non_ingredients


def filter_ner_column(df_cleaned, valid_phrases, identified_non_ingredients):
    """
    Filters the 'NER' column in the DataFrame by removing non-ingredient nouns.
    
    Args:
    - train_df (pd.DataFrame): The DataFrame that contains the 'NER' column.
    - valid_phrases (list): List of valid noun phrases to filter.
    - identified_non_ingredients (list): List of identified non-ingredient words/phrases.

    Returns:
    - train_df (pd.DataFrame): The DataFrame with an additional 'filtered_NER' column.
    """
    # Filter out non-ingredients from the valid phrases
    filtered_nouns = [noun for noun in valid_phrases if noun not in identified_non_ingredients]

    # Create a set for faster membership checking
    filtered_nouns_set = set(filtered_nouns)

    # Directly filter the NER column by checking if labels are in the filtered nouns set
    df_cleaned['filtered_NER'] = df_cleaned['NER'].apply(
        lambda labels: [label for label in labels if label in filtered_nouns_set]
    )

    return df_cleaned