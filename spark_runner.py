# Run a spark NLP model
def spark_runner(model: str = 'small') -> None:
    '''
        Description: This function is used to load the data from the dataset and process it to identify the named entities.
        Output: Inside a file spark_small.csv, the data is stored in the following
                format: text, type
        Model Used: Spark Small/Medium/Large
        Default: Spark Small
    '''
    # Load the spark model
    model_type = {
        'small': 'onto_recognize_entities_sm',
        # 'con-base': 'ner_conll_roberta_base',
    }
    model_name = model_type[model]
    sparknlp.start(gpu=False)
    # print(colored('Spark model: {model_name}', 'green').format(model_name=model_name))
    pipeline = PretrainedPipeline(model_name, lang='en')
    del model_type
    # Write the data to a file
    filename = f'./data/spark_{model}.csv'
    print(colored(f'[*] Writing data to {filename}', 'yellow'))
    with open(filename, 'w') as f:
        for df in pd.read_csv('./data.csv', chunksize=1000):
            # print(type(df))
            # print(df)
            # print(df.head())
            data = df['text'].values.tolist()
            del df
            for text in data:
                annotations = pipeline.fullAnnotate(u'{}'.format(text))
                for annotation in annotations:
                    # print(annotation)
                    # print(type(annotation))
                    for entity in annotation['entities']:
                        # print(entity)
                        type_of_result = entity.metadata.__getitem__('entity')
                        if type_of_result == 'PERSON' or type_of_result == 'ORG':
                            f.write(f'{entity.result},{type_of_result}\n')
                        del entity
                        continue
                del annotations
                continue
            continue
            del data
            continue
    return
