#!venv/bin/python3
# -*- coding: utf-8 -*-
#!pip install -r requirements.txt

'''
  Author: Avanish Gupta
	Date: August 3rd, 2021
	Description: This script is used to load the data from the dataset and process it to identify the named entities.
	Task: Implement Spark Sessions
'''
# Import the required libraries
import sparknlp  # Spark NLP Module
import os  # To create the folder structure
from termcolor import colored  # Color the output
import pandas as pd  # To read the csv file
from data import Data  # Import the data class
from matplotlib import pyplot as plt
import timeit  # To calculate the time taken to run the code
from memory_profiler import profile  # To measure memory usage
# Import the pretrained pipeline from SparkNLP
from sparknlp.pretrained import PretrainedPipeline
from multiprocessing import Process
from joblib import Parallel, delayed    # Parallel Processing


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
        for df in pd.read_csv('./data.csv'):
            print(type(df))
            print(df)
            print(df.head())
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

#	Process a function


def process_model(f) -> None:
    '''
        Description: This function is used to run the NLP model on the dataset.
        Input: Function name
        Output: None
    '''

    # Run the function
    with open('./results.csv', 'a') as result_file:
        # Calcuate the time
        start_time = timeit.default_timer()
        print('Started {f}'.format(f=f))
        try:
            f['function'](**f['parameters'])
        except Exception as e:
            print(e)
        elapsed = timeit.default_timer() - start_time
        print(
            colored(f'[*] {f["function"]} completed in {elapsed} seconds', 'green'))
        # Calculate the number of tags
        filename = f'./data/{f["function"].__name__[:-7]}_{f["parameters"]["model"]}.csv'

        # Store the results to the target file
        with open(filename, 'r') as file:
            # Get the number of tags extracted, and write the (package, model, time, number_of_tags)
            num_tags = len(file.readlines())
            print(
                f"{f['function'].__name__[:-7]}, {f['parameters']['model']}, {elapsed}, {num_tags}", file=result_file)
    return

# Run the functions and calculate their execution time


@profile
def time_counter(functions: list) -> None:
    '''
        Description: This function is used to time the execution of the functions  passed as an argument.
        Input: List of functions
        Output: None
    '''
    # Print the runtime of the functions
    # Parallel(n_jobs=4)(delayed(process_model)(f) for f in functions)
    for f in functions:
        process_model(f)
        # p1 = Process(process_model(f))
        # p1.start()
        # del f
        continue
    return

# Drive the program


def handle_data_folder():
    '''
        Description: This function is used to handle the data folder.
        Input: None
        Output: None
    '''
    # Create the data folder if it does not exist
    if os.path.isdir('./data'):
        print('Folder structure already exists')
    else:
        os.makedirs('./data/', exist_ok=True)
    return


def plot_results(file: str = './results.csv') -> None:
    '''
        Description: This function is used to plot the results.
        Input: File name
        Output: None
    '''
    # Read the data
    df = pd.read_csv(file)
    # Print the head
    # print(df.head())
    # Get mean based on Library and Model
    data_to_be_plotted = df.groupby(['Library', 'Model']).mean()
    print(data_to_be_plotted)
    data_to_be_plotted.sort_values(by=['Time', 'Tags'], inplace=True)
    mean_df = data_to_be_plotted.plot(kind='bar', rot=30, figsize=(10, 12))
    # Save the plot
    # Chnage the x labels of each plot
    mean_df.set_xlabel('Models')
    mean_df.get_figure().savefig('./results.png')
    # Print the mean
    print(mean_df)
    df.groupby(['Library']).mean().plot()
    return


def main():
    '''
        Description: Driver Function to call other functions
    '''

    # Handle the data folder
    handle_data_folder()

    # Time the execution of the functions
    functions = [
        # {'function': spacy_runner, 'parameters': {'model': 'small'}},
        # {'function': spacy_runner, 'parameters': {'model': 'medium'}},
        # {'function': spacy_runner, 'parameters': {'model': 'large'}},
        # {'function': flair_runner, 'parameters': {'model':'small'}},
        # {'function': flair_runner, 'parameters': {'model':'fast'}},
        # {'function': flair_runner, 'parameters': {'model':'large'}},
        # {'function': flair_runner, 'parameters': {'model':'ontonotes'}},
        # {'function': flair_runner, 'parameters': {'model':'ontonotes-fast'}},
        # {'function': stanza_runner, 'parameters': {'model':'fast'}},
        {'function': spark_runner, 'parameters': {'model': 'small'}},
        # {'function': spark_runner, 'parameters': {'model':'con-base'}},
        # {'function': flair_runner, 'parameters': {'model':'ontonotes-large'}},
    ]
    time_counter(functions)
    plot_results()
    return


# Enforce the main function
if __name__ == '__main__':
    main()
    exit(0)
