import spacy

nlp = spacy.load("./output/model-best")
doc = nlp("1 can cream-style corn")
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)