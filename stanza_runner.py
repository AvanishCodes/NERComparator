# Run a stanza based NLP model
def stanza_runner(model: str = 'small') -> None:
    '''
        Description: This function is used to load the data from the dataset and process it to identify the named entities.
        Output: Inside a file stanza_small.csv, the data is stored in the following
                format: text, type
        Model Used: Stanza Small/Medium/Large
        Default: Stanza Small
    '''
    # Load the stanza model
    nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')

    # Write the data to a file
    filename = f'./data/stanza_{model}.csv'
    with open(filename, 'w') as f:
        for df in pd.read_csv('./data.csv', chunksize=32):
            data = df['text'].values.tolist()
            del df
            for text in data:
                # Process the text
                doc = nlp(u'{}'.format(text))
                del text
                for ent in doc.entities:
                    if ent.type == 'PERSON' or ent.type == 'ORG':
                        f.write(f'{ent.text},{ent.type}\n')
                    del ent
                    continue
                del doc
                continue
            continue
    return
