# Run a spaCy NLP model
def spacy_runner(model: str = 'medium') -> None:
    '''
        Description: This function is used to load the data from the dataset and process it to identify the named entities.
        Output: Inside a file spacy_large.csv, the data is stored in the following
                format:	token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop
        Model Used: Spacy Small/Medium/Large
        Default: Spacy Medium
    '''

    # Determine and Load the spacy model
    model_name = 'en_core_web_sm' if model == 'small' else 'en_core_web_lg' if model == 'large' else 'en_core_web_md'
    nlp = spacy.load(model_name)
    del model_name
    # Write the data to a file
    filename = f'./data/spacy_{model}.csv'
    with open(filename, 'w') as f:
        for df in pd.read_csv('./data.csv', chunksize=32):
            data = df['text'].values.tolist()
            del df
            for text in data:
                # Process the text
                doc = nlp(u'{}'.format(text))
                for token in doc.ents:
                    if token.label_ == 'PERSON' or token.label_ == 'ORG' or token.label_ == 'PER':
                        f.write(f'{token.text},{token.label_}\n')
                    del token
                    continue
                del text
                del doc
                continue
            del data
            continue
    return
