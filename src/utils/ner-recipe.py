import spacy
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
import matplotlib.pyplot as plt
from tabulate import tabulate
from spacy.scorer import Scorer


def save_training_data(data, output_file):
    nlp = spacy.blank("en")
    doc_bin = DocBin()
    for text, annotations in data:
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annotations["entities"]:
            span = doc.char_span(start, end, label=label)
            if span is not None:
                ents.append(span)
        doc.ents = ents
        doc_bin.add(doc)
    doc_bin.to_disk(output_file)

def test_ner(recipe, nlp):
    predicted_ingredients = set()
    predicted_quantities = set()
    doc = nlp(recipe)
    
    for ent in doc.ents:
        if ent.label_ == 'INGREDIENT':
            predicted_ingredients.add(ent.text)
        elif ent.label_ == 'QUANTITY':
            predicted_quantities.add(ent.text)
    
    return predicted_ingredients, predicted_quantities
    

def test_and_save_models(test_df, test_ner, model_count=3, column_name="test_recipe"):
    # Iterate over the models, test them, and save results
    for i in range(1, model_count + 1):
        model_path = f"ner_model_{i}"
        print(f"Testing model: {model_path}")
        
        # Apply the test_ner function to the test data for each model
        test_df[f'predicted_ingredients_model_{i}'], test_df[f'predicted_quantities_model_{i}'] = zip(
            *test_df[column_name].apply(lambda x: test_ner(x, model_path))
        )
        
        # Save results to CSV for each model
        test_df[f'predicted_ingredients_model_{i}'].to_csv(f'predicted_ingredients_model_{i}.csv', index=False)
        print(f"Results saved for model {i}.")

    return test_df

def evaluate_model(test_data, actual_data, nlp):
    predicted_data = [(text, {"entities": []}) for text in test_data]
    
    # Generate predictions for all texts
    for j, (text, annotations) in enumerate(predicted_data):
        doc = nlp(text)
        entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        predicted_data[j] = (text, {"entities": entities})

    examples = []
    for actual, predicted in zip(actual_data, predicted_data):
        text = actual[0]
        actual_anns = actual[1]
        predicted_anns = predicted[1]

        doc_actual = nlp.make_doc(text)
        example = Example.from_dict(doc_actual, actual_anns)

        doc_predicted = nlp.make_doc(text)
        predicted_spans = []
        for start, end, label in predicted_anns["entities"]:
            span = doc_predicted.char_span(start, end, label=label)
            if span is not None:
                predicted_spans.append(span)
        doc_predicted.ents = predicted_spans
        example.predicted = doc_predicted
        examples.append(example)

    scorer = Scorer()
    metrics = scorer.score(examples)
    
    return metrics
    
def print_metrics(metrics):
    print("Model Metrics:")
    print(f"Token Accuracy: {metrics['token_acc']}")
    print(f"Token Precision: {metrics['token_p']}")
    print(f"Token Recall: {metrics['token_r']}")
    print(f"Token F1 Score: {metrics['token_f']}")
    print(f"Entity Precision: {metrics['ents_p']}")
    print(f"Entity Recall: {metrics['ents_r']}")
    print(f"Entity F1 Score: {metrics['ents_f']}")

    # Prepare data for entity metrics in a table format
    entity_data = []
    for entity, scores in metrics['ents_per_type'].items():
        entity_data.append([entity, scores['p'], scores['r'], scores['f']])
    
    # Print the table
    print(tabulate(entity_data, headers=["Entity", "Precision", "Recall", "F1 Score"], tablefmt="grid"))
    print("\n")

def save_metrics_and_plot(parameter_sets, all_metrics):
    # Checking if the file exists and if it has been written before
    file_exists = os.path.isfile("metrics_summary.csv")
    
    # Open the file and append metrics
    with open("metrics_summary.csv", "a") as f:
        # Only write the header if the file doesn't exist or is empty
        if not file_exists:
            f.write("model,parameter,precision,recall,f1_score\n")

        # Write the metrics for each parameter set
        for i, metric in enumerate(all_metrics):
            f.write(f"model_{i + 1},{parameter_sets[i]},{metric['ents_p']},{metric['ents_r']},{metric['ents_f']}\n")

    # Collect metrics for visualization
    precisions = [metric["ents_p"] for metrics in all_metrics]
    recalls = [metric["ents_r"] for metrics in all_metrics]
    f1_scores = [metric["ents_f"] for metrics in all_metrics]
    
    return precisions, recalls, f1_scores

def plot_metrics(precisions, recalls, f1_scores):
    """
    Plots Precision, Recall, and F1-Score for each model.

    Args:
    precisions (list): List of precision scores for each model.
    recalls (list): List of recall scores for each model.
    f1_scores (list): List of F1-scores for each model.
    """
    plt.figure(figsize=(10, 6))
    x = range(1, len(precisions) + 1)

    # Plot Precision, Recall, and F1-Score
    plt.plot(x, precisions, marker='o', label="Precision")
    plt.plot(x, recalls, marker='o', label="Recall")
    plt.plot(x, f1_scores, marker='o', label="F1-Score")

    # Set x-ticks and labels
    plt.xticks(x, [f"Model {i}" for i in x])

    # Add labels and title
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.title("NER Model Evaluation Metrics")

    # Display the legend and grid
    plt.legend()
    plt.grid()

    # Show the plot
    plt.show()