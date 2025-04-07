import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,roc_curve,auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, label_binarize
import joblib
import ast
from collections import Counter

def create_interactive_table(data, title, max_rows=None):
  
    if max_rows is not None:
        df = data.head(max_rows)
    else:
        df = data

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df.columns),
            fill_color='paleturquoise',
            align='left'
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color='lavender',
            align='left'
        )
    )])
    fig.update_layout(title=title)
    return fig
def create_info_df(df):
    info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = df[col].count()
        info.append([col, dtype, non_null])

    info_df = pd.DataFrame(info, columns=['Column', 'Dtype', 'Non-Null Count'])

    total_row = pd.DataFrame({
        'Column': [f'Total ({len(df)} rows)'],
        'Dtype': [''],
        'Non-Null Count': ['']
    })
    
    return pd.concat([info_df, total_row], ignore_index=True)

def create_missing_values_df(train_df, test_df):
    train_missing = train_df.isnull().sum()
    test_missing = test_df.isnull().sum()
    
    missing_data = pd.DataFrame({
        'Column': train_missing.index,
        'Training Data Missing': train_missing.values,
        'Test Data Missing': test_missing.reindex(train_missing.index).fillna(0).astype(int)
    })
    
    missing_data['Training Data Missing (%)'] = (missing_data['Training Data Missing'] / len(train_df) * 100).round(2)
    missing_data['Test Data Missing (%)'] = (missing_data['Test Data Missing'] / len(test_df) * 100).round(2)
    
    return missing_data.sort_values('Training Data Missing', ascending=False)

def create_unique_cuisines_df(train_df):
    if 'cuisine' in train_df.columns:
        cuisine_counts = train_df['cuisine'].value_counts().reset_index()
        cuisine_counts.columns = ['Cuisine', 'Count']
        cuisine_counts['Percentage'] = (cuisine_counts['Count'] / len(train_df) * 100).round(2)
        
        return cuisine_counts.sort_values('Count', ascending=False)
    else:
        raise ValueError("The 'cuisine' column is not present in the training data.")

# try:
#     unique_cuisines_df = create_unique_cuisines_df(train_df)
#     unique_cuisines_table = create_interactive_table(unique_cuisines_df, "Unique Cuisines in Training Data")
#     unique_cuisines_table.show()
# except ValueError as e:
#     print(e)

def create_common_ingredients_df(train_df, top_n=10):
    if 'ingredients' in train_df.columns:
        all_ingredients = [item for sublist in train_df['ingredients'] for item in sublist]
        ingredient_freq = Counter(all_ingredients)
        
        top_ingredients = ingredient_freq.most_common(top_n)
        
        data = {
            'Rank': list(range(1, top_n + 1)),
            'Ingredient': [item[0] for item in top_ingredients],
            'Frequency': [item[1] for item in top_ingredients],
            'Percentage': [round((item[1] / len(all_ingredients) * 100), 2) for item in top_ingredients]
        }
        
        return pd.DataFrame(data)
    else:
        raise ValueError("The 'ingredients' column not in the training data.")

# try:
#     common_ingredients_df = create_common_ingredients_df(train_df, top_n=10)
#     common_ingredients_table = create_interactive_table(common_ingredients_df, "Top 10 Most Common Ingredients")
#     common_ingredients_table.show()
# except ValueError as e:
#     print(e)

def create_avg_ingredients_df(train_df):
    if 'cuisine' in train_df.columns and 'ingredients' in train_df.columns:
        cuisine_ingredient_count = train_df.groupby('cuisine')['ingredients'].apply(lambda x: np.mean([len(i) for i in x]))
        
        df = cuisine_ingredient_count.reset_index()
        df.columns = ['Cuisine', 'Average Ingredients']
        
        df = df.sort_values('Average Ingredients', ascending=False)
        
        df['Average Ingredients'] = df['Average Ingredients'].round(2)
        
        df['Rank'] = range(1, len(df) + 1)
        
        df = df[['Rank', 'Cuisine', 'Average Ingredients']]
        
        return df
    else:
        raise ValueError("The 'cuisine' or 'ingredients' column not in the training data.")

# try:
#     avg_ingredients_df = create_avg_ingredients_df(train_df)
#     avg_ingredients_table = create_interactive_table(avg_ingredients_df, "Average Number of Ingredients by Cuisine")
#     avg_ingredients_table.show()
# except ValueError as e:
#     print(e)

def process_set_string(set_str):   
    try:
        ingredients_set = ast.literal_eval(set_str)
      
        return ' '.join(ingredients_set)
    except (ValueError, SyntaxError):
        return ""  # Return empty string if conversion fails
    
def create_model_performance_df(model, X, y, model_name):
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df['model'] = model_name
    return df.reset_index().rename(columns={'index': 'metric'})

def plot_roc_curves(y_true, y_probs, model_names):
    
    y_bin = label_binarize(y_true, classes=np.unique(y_true))
    n_classes = y_bin.shape[1]

    # Computing ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_probs[0][:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fig = go.Figure()
    for i, model_name in enumerate(model_names):
        fig.add_trace(go.Scatter(x=fpr[0], y=tpr[0],
                                 mode='lines',
                                 name=f'{model_name} (AUC = {roc_auc[0]:.2f})'))

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                             mode='lines',
                             name='Random Guess',
                             line=dict(dash='dash')))

    fig.update_layout(title='Receiver Operating Characteristic (ROC) Curve',
                      xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate')
    return fig
#Ensemble Probabilities
def plot_ensemble_probabilities(ensemble_probs, label_encoder):
    df = pd.DataFrame(ensemble_probs, columns=label_encoder.classes_)
    df_melted = df.melt(var_name='Cuisine', value_name='Probability')
    
    fig = go.Figure()
    for cuisine in label_encoder.classes_:
        fig.add_trace(go.Box(y=df[cuisine], name=cuisine))
    
    fig.update_layout(title='Ensemble Model - Prediction Probabilities by Cuisine',
                      yaxis_title='Probability',
                      boxmode='group')
    fig.show()

#Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    fig = go.Figure(data=go.Heatmap(z=cm, x=classes, y=classes, colorscale='Viridis'))
    fig.update_layout(title='Ensemble Model - Confusion Matrix',
                      xaxis_title='Predicted label',
                      yaxis_title='True label')
    fig.show()

#Classification Report
def plot_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Metric'] + list(df.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[df.index] + [df[col] for col in df.columns],
                   fill_color='lavender',
                   align='left'))
    ])
    fig.update_layout(title='Ensemble Model - Classification Report')
    fig.show()
#Model Agreement Heatmap
def model_agreement(probs1, probs2):
    return np.mean(np.argmax(probs1, axis=1) == np.argmax(probs2, axis=1))

def predict_cuisine(ingredients, rf_pipeline, gb_pipeline, svc_pipeline, label_encoder):
    input_str = ' '.join(ingredients)
    rf_probs = rf_pipeline.predict_proba([input_str])[0]
    gb_probs = gb_pipeline.predict_proba([input_str])[0]
    svc_probs = svc_pipeline.predict_proba([input_str])[0]
    ensemble_probs = (rf_probs + gb_probs + svc_probs) / 3
    predicted_class = np.argmax(ensemble_probs)
    predicted_cuisine = label_encoder.inverse_transform([predicted_class])[0]
    confidence_score = ensemble_probs[predicted_class]
    return predicted_cuisine, confidence_score, ensemble_probs

# Function to predict cuisine in batches
def predict_cuisine_batch(ingredients_list, rf_pipeline, gb_pipeline, svc_pipeline, label_encoder):
    rf_probs = rf_pipeline.predict_proba(ingredients_list)
    gb_probs = gb_pipeline.predict_proba(ingredients_list)
    svc_probs = svc_pipeline.predict_proba(ingredients_list)
    ensemble_probs = (rf_probs + gb_probs + svc_probs) / 3
    predicted_classes = np.argmax(ensemble_probs, axis=1)
    predicted_cuisines = label_encoder.inverse_transform(predicted_classes)
    confidence_scores = ensemble_probs[np.arange(len(ensemble_probs)), predicted_classes]
    return predicted_cuisines, confidence_scores