import pandas as pd
import numpy as np
import spacy
from spacy.language import Language
from spacy.tokens import Span, DocBin, Doc
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher
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

# Ingredient extractor component
@Language.component("ingredient_extractor")
def ingredient_extractor(doc, ner_tags):
    nlp = spacy.blank("en")
    patterns = create_patterns(nlp, ner_tags)
    ingredient_matcher = initialize_phrase_matcher(nlp, patterns)
    spans = extract_ingredient_spans(doc, ingredient_matcher)
    unique_spans = filter_overlapping_spans(doc, spans)
    update_doc_entities(doc, unique_spans)
    return doc

def create_patterns(nlp, ner_tags):
    terms = {}
    patterns = []
    for tags in ner_tags:
        for tag in tags:
            if tag not in terms and tag != 'mix':
                terms[tag] = {'label': 'INGREDIENT'}
                patterns.append(nlp(tag))
    return patterns

def initialize_phrase_matcher(nlp, patterns):
    ingredient_matcher = PhraseMatcher(nlp.vocab)
    ingredient_matcher.add("INGREDIENT", None, *patterns)
    return ingredient_matcher

def extract_ingredient_spans(doc, ingredient_matcher):
    matches = ingredient_matcher(doc)
    return [Span(doc, start, end, label='INGREDIENT') for match_id, start, end in matches]

def filter_overlapping_spans(doc, spans):
    quantity_spans = [ent for ent in doc.ents if ent.label_ == "QUANTITY"]
    unique_spans = []
    for span in spans:
        if not any(span.start < quantity.end and span.end > quantity.start for quantity in quantity_spans):
            unique_spans.append(span)
    return spacy.util.filter_spans(unique_spans)

def update_doc_entities(doc, unique_spans):
    new_ents = [ent for ent in doc.ents if ent.label_ != "INGREDIENT"]
    doc.ents = new_ents + unique_spans


