python spacy-train.py
python verify-spacy.py
python -m spacy train config.cfg --paths.train ner_training_data.spacy --paths.dev ner_training_data.spacy --output ./output
python test-model.py