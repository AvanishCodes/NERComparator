#!venv/bin/python3
# -*- coding: utf-8 -*-
#!pip install -r requirements.txt

'''
  Author: Avanish Gupta
	Date: August 3rd, 2021
	Description: This script is used to load the data from the dataset and process it to identify the named entities.
'''
# Import the required libraries
import os  # To create the folder structure
from termcolor import colored  # Color the output
from flair_runner import flair_runner
from spacy_runner import spacy_runner
from spark_runner import spark_runner
from stanza_runner import stanza_runner
import pandas as pd  # To read the csv file
from matplotlib import pyplot as plt
import timeit  # To calculate the time taken to run the code
from memory_profiler import profile  # To measure memory usage
# Import the pretrained pipeline from SparkNLP
from multiprocessing import Process
from joblib import Parallel, delayed    # Parallel Processing


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
        {'function': spacy_runner, 'parameters': {'model': 'small'}},
        {'function': spacy_runner, 'parameters': {'model': 'medium'}},
        {'function': spacy_runner, 'parameters': {'model': 'large'}},
        # {'function': flair_runner, 'parameters': {'model':'small'}},
        # {'function': flair_runner, 'parameters': {'model':'fast'}},
        # {'function': flair_runner, 'parameters': {'model':'large'}},
        # {'function': flair_runner, 'parameters': {'model':'ontonotes'}},
        # {'function': flair_runner, 'parameters': {'model':'ontonotes-fast'}},
        # {'function': stanza_runner, 'parameters': {'model':'fast'}},
        # {'function': spark_runner, 'parameters': {'model': 'small'}},
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
