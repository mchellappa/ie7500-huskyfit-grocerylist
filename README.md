#### Project Proposal: Ingredient Extraction and Categorization 
 
#### IE 7500: Applied Natural Language Processing in Engineering - Group A 

###### Divya Maheshkumar 

###### Muthu Chellappa 

###### SenthilKumaran Ramanathan   

#### Title 

# Ingredient Intelligence

This project combines Pre-Processing, Exploratory data analysis (EDA), Named Entity Recognition (NER) model training, and cuisine classification to extract ingredients from recipes and predict their cuisines. It processes raw recipe data, trains a SpaCy NER model to identify ingredients and quantities, and uses an ensemble machine learning model to classify cuisines based on those ingredients.

## Table of Contents
1. Project Overview
2. Requirements
3. File Descriptions
   - Data Preprocessing and EDA ('preprocessing-notebook.ipynb')
   - NER Model Training ('ner-notebook.ipynb')
   - Cuisine Prediction Classification ('cuisine_prediction_classification.ipynb')
4. Usage
5. Outputs
6. Notes

## Project Overview
The project is a three-phase pipeline for analyzing and classifying recipe data:
1. Data Preprocessing and EDA: Cleans and normalizes a raw recipe dataset ('recipes_data.csv'), preparing it for NER tasks by removing noise and standardizing text.
2. NER Model Training: Trains a SpaCy NER model to extract 'INGREDIENT' and 'QUANTITY' entities using span-based annotation(rule based matching) from recipe text, enhancing ingredient identification.
3. Cuisine Prediction Classification: Uses an ensemble of Random Forest, Gradient Boosting, and Linear SVC models to predict cuisines from identified ingredients, validated on test samples and applied to a large dataset.

This end to end workflow transforms unstructured recipe text into structured data (ingredients) and predicts cuisines, enabling applications like recipe categorization or recommendation systems.

## Requirements
- Python: 3.8+
- Libraries:
  - 'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn', 'wordcloud', 'ast' (EDA)
  - 'spacy', 'random' (NER)
  - 'plotly', 'joblib', 'tqdm'
- Install dependencies:
  '''pandas numpy scikit-learn matplotlib seaborn wordcloud spacy plotly joblib tqdm
  '''
- Datasets:
  - 'recipes_data.csv': Raw recipe data with 'title', 'ingredients', 'directions', 'NER'([Recipe Dataset](https://www.kaggle.com/datasets/wilmerarltstrmberg/recipe-dataset-over-2m/data)).   
  - 'train.json', 'test.json': Labeled data for cuisine classification ([Recipe Ingredients Dataset](https://www.kaggle.com/datasets/kaggle/recipe-ingredients-dataset/data)).

## File Descriptions

### Data Preprocessing and EDA
Purpose: Prepares raw recipe data for NER and downstream tasks through EDA and cleaning.

Key Components:
- Loads 'recipes_data.csv',sample 150000 records, splits into 80% training and 20% testing.
- Performs EDA (e.g., ingredient counts, NER label distribution) with visualizations (word clouds, bar plots).
- Cleans data by removing nulls, duplicates, and rows with good NER accuracy.
- Normalizes text (e.g., fractions to decimals, standardized units) into 'normalized_combined'.


### NER Model Training
Purpose: Trains a SpaCy NER model to extract ingredients and quantities from preprocessed recipe text.

Key Components:
- Loads 'preprocessed_train_data.csv' and 'test_data.csv'.
- Sets up a SpaCy pipeline with custom 'quantity_extractor' and 'ingredient_extractor' components.
- Trains the NER model on training data (10 iterations) and saves it as 'ner_model'.
- Applies the model to test data, saving predictions as 'predicted_ingredients.csv'.
- Tests on sample recipes, displaying entities with 'displacy'.

Inputs: 'preprocessed_train_data.csv', 'test_data.csv'  
Outputs: 'training_data.spacy', 'ner_model', 'predicted_ingredients.csv', console predictions


### Cuisine Prediction Classification
Purpose: Predicts cuisines from ingredients using an ensemble model, validated on test cases and scaled to a full dataset.

Key Components:
- Tests the ensemble on 5 hardcoded examples, displaying results in an interactive table.
- Predicts cuisines for 'predicted_ingredients.csv' (2M recipes) in batches, with visualizations.
- 'cuisine_prediction_classification.ipynb': Trains Random Forest, Gradient Boosting, and Linear SVC models on 'train.json', evaluates on 'test.json', and applies to 'predicted_ingredients.csv'.

Inputs: 
- 'train.json', 'test.json' (training/evaluation)
- 'predicted_ingredients.csv' (NER output)
- Model files: 'randomforest_model.pkl', 'gradientboosting_model.pkl', 'linearsvc_model.pkl', 'label_encoder.pkl'

Outputs: 
- Model files, 'ensemble_test_predictions.csv', 'ensemble_recipes_with_cuisines.csv'
- Interactive tables, bar charts


## Usage

### Running Data Preprocessing and EDA
   Place 'recipes_data.csv' in the working directory.

### Running NER Model Training
  Ensure 'preprocessed_train_data.csv' and 'test_data.csv' are present.

### Running Cuisine Prediction Classification
  Place 'train.json', 'test.json', and 'predicted_ingredients.csv' in the directory.

- Test Full Dataset:
  Ensure 'predicted_ingredients.csv' and model files are present.


## Outputs
- Preprocessing and EDA:
  - Files: 'preprocessed_train_data.csv' ('normalized_combined', 'NER'), 'test_data.csv' ('test_recipe', 'Actual_NER').
  - Visualizations: Word cloud, bar plots, scatter plots, histogram.
  - Console: Dataset stats, cleaning progress.
- NER Training:
  - Files: 'ner_model', 'predicted_ingredients.csv'.

- Cuisine Prediction:
  - Main: Model files, 'ensemble_test_predictions.csv', 'ensemble_recipes_with_cuisines.csv', extensive visualizations.
  - Test 5 Samples: Table (e.g., 'rice, soy sauce -> chinese, 0.483235'), bar charts.
  - Test Full Dataset: 'recipes_with_predicted_cuisines.csv', batch CSVs, table, bar chart, histogram, stats.


## Notes
- Pipeline Flow: Run 'preprocessing-notebook.ipynb' first, then 'ner-notebook.ipynb', and finally 'cuisine_prediction_classification.ipynb' and its test scripts. 
- NER: Increase iterations (e.g., 20–50) for better accuracy.
- Cuisine Prediction: Tweak 'batch_size' (default 10k) for full dataset processing.
- Visualization: Interactive plots (e.g., Plotly tables, Matplotlib figures),not all vizulizations are rendered by git, Please download and load the notebookfile in Juypiter notebook.
- We trained the model using a subset of 150,000 records from the dataset.This decision was made due to resource constraints, including computational power. 
- While processing the entire dataset would be ideal, we ensured that the selected subset is representative of the overall data distribution to maintain the integrity and reliability of the model's performance.

##### Project Repository Structure
```
/ie7500-huskyfit-grocerylist
│
├── src/
│   ├── api/v1/endpoints/recipe.py # Endpoint to generate ingredients and cuisine using the model
│   ├── input/                # Raw and preprocessed datasets
│   │   ├── recipes_data.csv
│   │   ├── train.json
│   │   ├── test.json
│   │   ├── preprocessed_train_data.csv
│   │   └── test_data.csv
│   ├── output/               # Model predictions and results
│   │   ├── predicted_ingredients.csv
│   │   ├── ensemble_test_predictions.csv
│   │   ├── ensemble_recipes_with_cuisines.csv
│   │   └── recipes_with_predicted_cuisines.csv
│   ├── models/               # Trained models and artifacts
│   │   ├── ner_model/
│   │   │   ├── model-best/
│   │   │   └── model-last/
│   │   ├── randomforest_model.pkl
│   │   ├── gradientboosting_model.pkl
│   │   ├── linearsvc_model.pkl
│   │   └── label_encoder.pkl
│   ├── visualizations/       # Generated visualizations
│   │   ├── wordclouds/
│   │   ├── bar_plots/
│   │   └── interactive_tables/
│   ├── main.py               # Entry point for running the pipeline
│   └── requirements.txt      # Dependencies
│
├── notebooks/                # Jupyter notebooks for each phase
│   ├── preprocessing-notebook.ipynb
│   ├── ner-notebook.ipynb
│   ├── cuisine_prediction_classification.ipynb
│   └── explore/20k/          # Exploratory analysis
│       ├── evaluation_code.ipynb
│       └── README.md
├── evaluation results/       # Model evaluation metrics and visualizations
│   ├── Evaluation metrics.png
│   ├── model1 results.png
│   ├── model2 results.png
│   ├── model3 results.png
│   └── model4 results.png
└── README.md
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd ie7500-huskyfit-grocerylist
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

## 

### Prerequisites

The Models might not be available, Please run the notebooks to generate the models and move them to the appropriate folder.

To run the *Ingredient Intelligence* application, execute the following command:

```
uvicorn src.main:app --reload
```
You can then send a POST request to the `/process-recipe` endpoint with the recipe text to receive processed quantities and ingredients.

## API Endpoint

- **POST /process-recipe**
  - Request Body: JSON object containing the recipe text.
  - Response: JSON object with extracted quantities and ingredients.

## License

This project is licensed under the MIT License.
