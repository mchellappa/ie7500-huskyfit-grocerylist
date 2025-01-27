import spacy
from spacy.tokens import DocBin
nlp = spacy.blank("en")

# Load the processed data
doc_bin = DocBin().from_disk("ner_training_data.spacy")
docs = list(doc_bin.get_docs(nlp.vocab))

for doc in docs[:5]:  # Print the first few examples
    print("Text:", doc.text)
    print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])