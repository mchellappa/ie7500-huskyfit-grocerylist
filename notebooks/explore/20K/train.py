import random
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
from spacy.tokens import DocBin
import os

# Load the training data
def train_model(max_iterations=10, 
                min_loss_improvement=5000, 
                output_dir="ner_model",
                compounding_factor_start=4.0,
                compounding_factor_stop=32.0,
                compounding_factor_increase=1.5,
                drop=0.5):
    print("Loading training data...")
    nlp = spacy.blank("en")  # Using a blank 'en' model
    db = DocBin().from_disk("training_data.spacy")
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

        #max_iterations = max_iterations  # Total iterations
        #min_loss_improvement = min_loss_improvement  # Minimum loss improvement to continue training
        prev_loss = None  # Track the previous loss

        #Parallelized minibatch to process training data in parallel
        for itn in range(max_iterations):
            random.shuffle(docs)
            losses = {}
            batches = minibatch(docs, size=compounding(compounding_factor_start, compounding_factor_stop, compounding_factor_increase))

            for batch in batches:
                # Examples from docs and update the model
                examples = [Example.from_dict(doc, {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]}) for doc in batch]

                nlp.update(examples, drop=drop, losses=losses)

            print(f"Iteration {itn}, Losses: {losses}")

            # Check for improvement in loss
            if prev_loss is not None:
                improvement = prev_loss - losses['ner']
                print(f"Improvement in this iteration: {improvement}")

                # Early stopping if the improvement is small
                if improvement < min_loss_improvement:
                    print(f"Stopping early due to small improvement.")
                    break

            # Update previous loss for next iteration
            prev_loss = losses['ner']

        # Save the trained model to disk
        nlp.to_disk(output_dir)
    # Ensure the function does not call itself recursively
    print("Training completed. Model saved to disk.")