import spacy

nlp = spacy.load("./output/model-best")
doc = nlp("½ pound andouille or jalapeno chicken sausages")
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)