import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import glob

# Load the training data from chunks
nlp = spacy.blank("en")
doc_bin = DocBin()

for chunk_file in glob.glob("ner/ner_training_data_chunk_*.spacy"):
    chunk_bin = DocBin().from_disk(chunk_file)
    for doc in chunk_bin.get_docs(nlp.vocab):
        doc_bin.add(doc)

docs = list(doc_bin.get_docs(nlp.vocab))

# Create a new entity recognizer
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")

# Add labels to the entity recognizer
for doc in docs:
    for ent in doc.ents:
        ner.add_label(ent.label_)

# Disable other pipelines during training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()
    for i in range(20):  # Number of iterations
        for doc in docs:
            example = Example.from_dict(doc, {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]})
            nlp.update([example], drop=0.5, sgd=optimizer)

# Save the model
nlp.to_disk("./output/model-best")