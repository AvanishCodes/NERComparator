from flair.data import Sentence  # Flair Module
from flair.models import SequenceTagger  # Tagger inside Flair module

# Flair testing based on model input


def flair_runner(model: str = 'fast') -> None:
    '''
        Description: This function is used to load the data from the dataset and process it to identify the named entities.
        Input: Model type
        Output: Inside a file spacy_large.csv, the data is stored in the following format:	
        Model Used: Flair English Model: large/small/fast/ontonotes/ontonotes-fast/ontonotes-large
    '''

    # Determine the flair model to be used
    model_type = {
        'small': 'ner-english',
        'large': 'ner-english-large',
        'fast': 'ner-english-fast',
        'ontonotes': 'ner-english-ontonotes',
        'ontonotes-fast': 'ner-english-ontonotes-fast',
        'ontonotes-large': 'ner-english-ontonotes-large'
    }
    # load tagger based on the model
    model_name = model_type[model]
    del model_type
    tagger = SequenceTagger.load(f'flair/{model_name}')

    # Determine the file to read into
    model_file = {
        'small': 'small',
        'large': 'large',
        'fast': 'fast',
        'ontonotes': 'ontonotes',
        'ontonotes-fast': 'ontonotes-fast',
        'ontonotes-large': 'ontonotes-large'
    }
    filename = f'./data/flair_{model_file[model]}.csv'
    del model_file
    # Write the data to a file
    with open(filename, 'w') as f:
        for df in pd.read_csv('./data.csv', chunksize=32):
            data = df['text'].values.tolist()
            for text in data:
                # Process the text
                sentence = Sentence(text)
                # predict NER tags
                tagger.predict(sentence)
                del text
                # Save the found tags in a file
                for entity in sentence.get_spans('ner'):
                    if entity.tag == 'PERSON' or entity.tag == 'ORG' or entity.tag == 'PER':
                        print(entity.start_pos, entity.end_pos,
                              entity.text, entity.tag, sep=',', file=f)
                    del entity
                    continue
                del sentence
                continue
            continue
    return
