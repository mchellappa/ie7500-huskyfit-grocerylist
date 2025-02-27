#### Project Proposal: Ingredient Extraction and Categorization 

 
#### IE 7500: Applied Natural Language Processing in Engineering - Group A 

## Divya Maheshkumar 

## Muthu Chellappa 

## SenthilKumaran Ramanathan   

#### Title 

### Automated Ingredient Extraction and Categorization from Recipe Texts Using Natural Language Processing 

### Introduction 

Recipes are a rich source of culinary data, often containing unstructured text that includes ingredients, quantities, and preparation instructions. Extracting and categorizing ingredients from these texts can facilitate various applications, such as nutritional analysis, recipe personalization, and inventory management.  

### Problem Statement  

Manually extracting ingredients and categorizing them from recipe texts is labor-intensive, error-prone, and often inefficient. This challenge is compounded by the diverse formats and variations found in recipe texts, from differing terminology and units to inconsistencies in ingredient descriptions. An automated system utilizing Natural Language Processing techniques could vastly improve the accuracy and speed of this process. Such a system would enable more reliable recipe analysis, facilitate dietary planning, and support the growing field of culinary research, all while reducing human error. 

### Objectives 

## Ingredient Extraction: To develop a system that accurately extracts ingredient names from recipe texts. 

## Ingredient Categorization: To categorize extracted ingredients into meaningful groups such as ingredient types (e.g., spices, vegetables) or cuisines (e.g., Italian, Indian). 

## System Evaluation: To evaluate the system's performance and accuracy in real-world recipe datasets. 

### Dataset Selection  

We will use the Recipe Dataset (over 2M) Food dataset available on Kaggle. This dataset contains: 

https://www.kaggle.com/datasets/wilmerarltstrmberg/recipe-dataset-over-2m/data 

Over 2 million recipes 

Detailed information including ingredients, instructions, and nutritional facts 

A diverse range of cuisines and dietary categories 

The large volume and variety of recipes in this dataset make it ideal for training a robust NLP model for ingredient extraction and categorization. 

 

### Methodology 

## Data Labelling and Text Preprocessing 

Data Labelling for ingredients, quantities, measurement 

Clean and normalize recipe texts to ensure consistency in formats (e.g., removing special characters, handling abbreviations). 

Tokenization: break down text into words 

Lowercase: convert to lower case 

Lemmatization: reduce words to base form 

Remove stop words: remove words like 'a', 'the', 'of' unless they are useful for context 

 

### Ingredient Extraction 



 

### Execution 

Objective: To develop a NER Extraction model to select the Ingredients and Quantities from a given recipe text 

Dataset Prep: 

Download the dataset from Kaggle 

Clean and format the dataset by: 

Removing duplicates and null values. 

Standardizing ingredient text (e.g., lowercase, removing special characters). 

Converting string lists to Python lists using ast.literal_eval. 

Define the Quantities and Ingredient regex for the model 

 

Model Selection: In the initial phase, the focus will be on Spacy Rule based model. Using regex rules, will train the model and evaluate the model with the test dataset. Based on the results we will choose BERT or Deep learning. 

 

### Ingredient Categorization 

## Model Selection 



 

 

## Execution 

Objective: To develop a categorization model to classify ingredients into groups of either ingredient types or cuisines. 

Data Labeling: Correctly labeled with the appropriate category (e.g., "vegetable," "spice," or specific cuisine types like "Italian," "Indian," etc.) to form the ground truth for model training. 

Feature Extraction: 

Ingredient Representation: Convert the ingredient names into numerical representations that can be fed into machine learning models. Several techniques will be explored to identify the better fit for the task among the options 

TF-IDF (Term Frequency-Inverse Document Frequency): Capture the importance of each ingredient in relation to others by weighing its frequency across different recipes. 

Bag-of-Words (BoW): Treat each ingredient as a unique token in a collection of recipes to create a sparse vector representation. 

Embeddings: Pre-trained word embeddings (e.g., Word2Vec, GloVe) will be used to map ingredients to continuous vector spaces. 

 

Model Selection: In the initial phase, the focus will be on Gradient Boosting methods and Logistic Regression for ingredient classification. If additional model refinement is required, other techniques such as Random Forest, SVM, or Neural Networks may be explored in subsequent stages. 

 

Evaluation 

Measure the system's performance using precision, recall, and F1-score. 

Validate the ingredient extraction and categorization accuracy on a test dataset. 

Expected Outcomes   

An NLP model capable of accurately extracting ingredients, quantities, and units from recipe texts 

A classification system to categorize extracted ingredients/recipes into food groups 

A performance evaluation framework to assess the accuracy of extraction and categorization 

An interface to generate and extract ingredients from recipe text. 

# FastAPI Recipe Processing Application

This project is a FastAPI application that processes recipe text to extract quantities and ingredients using a SpaCy model. 

## Project Structure

```
Project
├── src
│   ├── main.py          # Entry point of the FastAPI application
│   ├── spacy_train.py   # Logic for initializing SpaCy model and processing text
│   └── types
│       └── index.ts     # TypeScript interfaces for type checking
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation
└── .env                  # Environment variables
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd my-fastapi-app
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the root directory and add any necessary configuration settings.

## Usage

To run the FastAPI application, execute the following command:

```
uvicorn src.main:app --reload
```

1. Preprocessing on both datasets
1.1 What preprocessing is required
2. NER for Ingredients
3. NER for Quantities
4. Model selection for Classification
4.1 Cuisine / Grocery group

You can then send a POST request to the `/process-recipe` endpoint with the recipe text to receive processed quantities and ingredients.

## API Endpoint

- **POST /process-recipe**
  - Request Body: JSON object containing the recipe text.
  - Response: JSON object with extracted quantities and ingredients.

## License

This project is licensed under the MIT License.