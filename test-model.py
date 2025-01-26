import spacy

nlp_ner = spacy.load("./output/model-best")
text = "1 c. peanut butter, 3/4 c. graham cracker crumbs, 1 c. melted butter, 1 lb. (3 1/2 c.) powdered sugar, 1 large pkg. chocolate chips"

doc = nlp_ner(text)

for ent in doc.ents:
    print(f"Entity: {ent.text}, Label: {ent.label_}")