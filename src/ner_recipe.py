import pandas as pd
import sys
import numpy as np
import spacy
from spacy.language import Language
from spacy.tokens import Span
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher

import random
from spacy.training import Example
from spacy.util import minibatch, compounding
from spacy.tokens import DocBin
import os
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print("Usage: python ner_recipe.py <input_file> <test_data_file>")
    sys.exit(1)

input_file = sys.argv[1]
test_data_file = sys.argv[2]

train_df = pd.read_csv(input_file)
print(f'Length of train data is {len(train_df)}')
train_df.head()
texts = train_df['normalized_combined'].tolist()
texts[0]
ner_tags = train_df['filtered_NER'].apply(eval).tolist()
ner_tags[0]
nlp = spacy.blank("en")

# Create a dictionary of terms
terms = {}
patterns = []

for tags in ner_tags:
    for tag in tags:
        if tag not in terms and tag!='mix':
            terms[tag] = {'label': 'INGREDIENT'}
            patterns.append(nlp(tag))

# Initialize the PhraseMatcher
ingredient_matcher = PhraseMatcher(nlp.vocab)
ingredient_matcher.add("INGREDIENT", None, *patterns)
patterns[:20]

nlp.analyze_pipes()

# Quantity extractor component
@Language.component("quantity_extractor")
def quantity_extractor(doc):
    # Extract quantities
    matcher = Matcher(nlp.vocab)
    pattern = [
        {"LIKE_NUM": True},  # Match numbers
        {"LIKE_NUM": True, "OP": "?"},  # Match the second optional number (e.g., 8)
        {"LOWER": {"IN": ["cup", "tablespoon", "teaspoon", "ounce", "pound", "gram", "kilogram", "package", "quart", "liter", "milliliter"]}}
    ]
    matcher.add("QUANTITY", [pattern])
    matches = matcher(doc)
    quantity_spans = [Span(doc, start, end, label="QUANTITY") for match_id, start, end in matches]

    filtered_spans = spacy.util.filter_spans(quantity_spans)
   # Filter out existing QUANTITY entities
    new_ents = [ent for ent in doc.ents if ent.label_ != "QUANTITY"]

    # Add the unique quantity spans to the new_ents list
    doc.ents = new_ents + filtered_spans  # Add unique quantity spans

    return doc
nlp.add_pipe("quantity_extractor", last=True)  # Quantity extractor runs first

# Ingredient extractor component
@Language.component("ingredient_extractor")
def ingredient_extractor(doc):
    # Extract ingredients after quantity extraction
    matches = ingredient_matcher(doc)
    spans = [Span(doc, start, end, label='INGREDIENT') for match_id, start, end in matches]

    unique_spans = []

    # Check if the ingredient overlaps with any quantity span before adding
    quantity_spans = [ent for ent in doc.ents if ent.label_ == "QUANTITY"]

    for span in spans:

        overlap_found = False

        # Check if the ingredient span overlaps with any quantity span
        for quantity in quantity_spans:
            if span.start < quantity.end and span.end > quantity.start:  # If overlap occurs
                overlap_found = True
                break  # Skip adding this ingredient if it overlaps with a quantity

        # Add ingredient span if there's no overlap with any quantity span
        if not overlap_found:
                unique_spans.append(span)


    # Resolve overlaps and filter out duplicate ingredient spans
    filtered_spans = spacy.util.filter_spans(unique_spans)
    new_ents = [ent for ent in doc.ents if ent.label_ != "INGREDIENT"]
    doc.ents = new_ents + filtered_spans  # Add unique ingredient spans, excluding overlapping ones

    return doc

nlp.add_pipe("ingredient_extractor", last=True)  # Ingredient extractor runs second

#Analyzing if Quantity and Ingredient Extractor components are added to pipe
nlp.analyze_pipes()
from spacy.tokens import DocBin

train_data = [(text, {"entities": []}) for text in texts]

for i, (text, annotations) in enumerate(train_data):
    doc = nlp(text)
    entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
    train_data[i] = (text, {"entities": entities})
# Access and print the 10th training data entry
example_data = train_data[10]
print(example_data)

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

save_training_data(train_data, 'training_data.spacy')



def train_ner_model(training_data_path, output_model_path, max_iterations, initial_lr, dropout_rate, min_loss_improvement, patience):
    # Load the training data
    nlp = spacy.blank("en")  # Using a blank 'en' model
    db = DocBin().from_disk(training_data_path)
    docs = list(db.get_docs(nlp.vocab))

    # Create the NER component and add it to the pipeline
    if 'ner' not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)

    # Add the labels to the NER component
    for doc in docs:
        for ent in doc.ents:
            ner.add_label(ent.label_)

    # Disable other pipes during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()

        prev_loss = None  # Track the previous loss
        no_improvement_counter = 0  # Counter for early stopping
        losses_history = []  # Track losses for plotting

        # Set an initial learning rate
        optimizer.learn_rate = initial_lr  # Set learning rate correctly

        for itn in range(max_iterations):
            random.shuffle(docs)
            losses = {}
            batches = minibatch(docs, size=compounding(4.0, 32.0, 1.5))

            for batch in batches:
                examples = [Example.from_dict(doc, {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]}) for doc in batch]
                nlp.update(examples, drop=dropout_rate, losses=losses)

            losses_history.append(losses['ner'])
            print(f"Iteration {itn}, Losses: {losses}")

            # Slower learning rate decay
            if itn % 3 == 0:  # Decay every 3 iterations
                new_lr = initial_lr * (0.5 ** (itn // 3))
                optimizer.learn_rate = new_lr
                print(f"Updated learning rate to: {new_lr}")

            # Check for improvement in loss
            if prev_loss is not None:
                improvement = prev_loss - losses['ner']
                print(f"Improvement in this iteration: {improvement}")

                if improvement < min_loss_improvement:
                    no_improvement_counter += 1
                    if no_improvement_counter >= patience:
                        print(f"Stopping early due to small improvement over {patience} iterations.")
                        break
                else:
                    no_improvement_counter = 0

            prev_loss = losses['ner']

        # Save the trained model to disk
        nlp.to_disk(output_model_path)

    return losses_history

# Iterate over three different parameter sets
parameter_sets = [
    {"max_iterations": 20, "initial_lr": 1e-2, "dropout_rate": 0.5, "min_loss_improvement": 5000, "patience": 3},
    {"max_iterations": 30, "initial_lr": 1e-3, "dropout_rate": 0.4, "min_loss_improvement": 3000, "patience": 5},
    {"max_iterations": 25, "initial_lr": 1e-4, "dropout_rate": 0.6, "min_loss_improvement": 2000, "patience": 4},
]

all_losses = []

for i, params in enumerate(parameter_sets):
    print(f"Training with parameter set {i + 1}: {params}")
    losses = train_ner_model(
        training_data_path="training_data.spacy",
        output_model_path=f"ner_model_{i + 1}",
        max_iterations=params["max_iterations"],
        initial_lr=params["initial_lr"],
        dropout_rate=params["dropout_rate"],
        min_loss_improvement=params["min_loss_improvement"],
        patience=params["patience"]
    )
    all_losses.append(losses)

# Plot the results
with open("training_losses.txt", "w") as f:
    for i, losses in enumerate(all_losses):
        f.write(f"Run {i + 1} Losses: {losses}\n")
        plt.plot(losses, label=f"Run {i + 1}")

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("NER Model Training Loss")
plt.legend()
plt.show()

# Bar chart for final losses
final_losses = [losses[-1] for losses in all_losses]
plt.bar(range(1, len(final_losses) + 1), final_losses)
plt.xlabel("Run")
plt.ylabel("Final Loss")
plt.title("Final Losses for Each Run")
plt.show()
test_df = pd.read_csv(test_data_file)
test_df.head()

test_df = test_df.rename(columns={'actual_NER': 'filtered_NER'})

# Load and test all three NER models
def test_ner(recipe, model_path):
    nlp = spacy.load(model_path)
    predicted_ingredients = set()
    predicted_quantities = set()
    doc = nlp(recipe)
    for ent in doc.ents:
        if ent.label_ == 'INGREDIENT':
            predicted_ingredients.add(ent.text)
        elif ent.label_ == 'QUANTITY':
            predicted_quantities.add(ent.text)
    return predicted_ingredients, predicted_quantities

# Iterate over the three models and test them
for i in range(1, 4):
    model_path = f"ner_model_{i}"
    print(f"Testing model: {model_path}")
    
    test_df[f'predicted_ingredients_model_{i}'], test_df[f'predicted_quantities_model_{i}'] = zip(
        *test_df['test_recipe'].apply(lambda x: test_ner(x, model_path))
    )

texts = test_df['test_recipe'].tolist()
ner_tags = test_df['predicted_ingredients'].tolist()

nlp = spacy.blank("en")

# Create a dictionary of terms
terms = {}
patterns = []

for tags in ner_tags:
    for tag in tags:
        if tag not in terms:
            terms[tag] = {'label': 'INGREDIENT'}
            patterns.append(nlp(tag))

# Initialize the PhraseMatcher
ingredient_matcher = PhraseMatcher(nlp.vocab)
ingredient_matcher.add("INGREDIENT", None, *patterns)

nlp.add_pipe("quantity_extractor", last=True)
nlp.add_pipe("ingredient_extractor", last=True)
nlp.analyze_pipes()
#Creating annotations for predicted ingredient values

from spacy.tokens import DocBin

predicted_data = [(text, {"entities": []}) for text in texts]

for i, (text, annotations) in enumerate(predicted_data):
    doc = nlp(text)
    entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
    predicted_data[i] = (text, {"entities": entities})

texts = test_df['test_recipe'].tolist()
ner_tags = test_df['filtered_NER'].apply(eval).tolist()

# Save results for each model
for i in range(1, 4):
    test_df[f'predicted_ingredients_model_{i}'].to_csv(f'predicted_ingredients_model_{i}.csv', index=False)

from spacy.tokens import DocBin

actual_data = [(text, {"entities": []}) for text in texts]

for i, (text, annotations) in enumerate(actual_data):
    doc = nlp(text)
    entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
    actual_data[i] = (text, {"entities": entities})

nlp = spacy.blank("en")



# Evaluate each model
from spacy.training import Example
from spacy.scorer import Scorer
import sys

for i in range(1, 4):
    print(f"Evaluating model: ner_model_{i}")
    model_path = f"ner_model_{i}"
    nlp = spacy.load(model_path)

    predicted_data = [(text, {"entities": []}) for text in test_df['test_recipe'].tolist()]
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
    precisions = []
    recalls = []
    f1_scores = []

    for i in range(1, 4):
        print(f"Evaluating model: ner_model_{i}")
        model_path = f"ner_model_{i}"
        nlp = spacy.load(model_path)

        predicted_data = [(text, {"entities": []}) for text in test_df['test_recipe'].tolist()]
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

        # Write metrics to a file with run parameters
        with open("metrics_summary.csv", "a") as f:
            if i == 1:
                f.write("model,parameter,precision,recall,f1_score\n")
            f.write(f"model_{i},{parameter_sets[i - 1]},{metrics['ents_p']},{metrics['ents_r']},{metrics['ents_f']}\n")

        
        print(f"Metrics for model {i}: {metrics}")
        
        # Collect metrics for visualization
        precision = metrics["ents_p"]
        recall = metrics["ents_r"]
        f1_score = metrics["ents_f"]
        
        # Append metrics to lists for plotting
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

# Plot precision, recall, and F1-score for each model
plt.figure(figsize=(10, 6))
x = range(1, 4)
plt.plot(x, precisions, marker='o', label="Precision")
plt.plot(x, recalls, marker='o', label="Recall")
plt.plot(x, f1_scores, marker='o', label="F1-Score")
plt.xticks(x, [f"Model {i}" for i in x])
plt.xlabel("Model")
plt.ylabel("Score")
plt.title("NER Model Evaluation Metrics")
plt.legend()
plt.grid()
plt.show()
