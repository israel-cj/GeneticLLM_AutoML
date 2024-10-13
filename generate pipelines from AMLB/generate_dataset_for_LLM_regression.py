# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 11:16:44 2021

@author: 20210595
"""

import pandas as pd
from datetime import datetime
from dateutil import parser
import numpy as np
import os
import glob
import pandas as pd
import ast
from ast import literal_eval
import pickle
import json


# plt.rcParams.update({'font.size': 10})

number_of_pipelines_to_consider = 10
path_use = os.getcwd()
files = glob.glob('**/*'+ ".log" , recursive=True)
files_csv = glob.glob('**/*'+ ".csv" , recursive=True)
# Create a dictionary with all the dataframes from the openml and a list as item
dataset_similarity = dict()
for file in files_csv:
    if 'results.csv' in file:
        file = file.replace(os.sep, '/')
        df = pd.read_csv(file)
        dataset_similarity[df['task'][0]] = []
        
for file in files:
    if "evaluations" in file:
        file = file.replace(os.sep, '/')
        df = pd.read_csv(file, sep=";")
        if len(df) > 0:
            try:
                # Get the name of the task
                name_task_path = file.split('/')
                for h in range(len(name_task_path)):
                    if name_task_path[h]=='logs':
                        this_task = name_task_path[h+1]
                # Delete when the csv has repeated name of columns, "keep=False" means delete all the duplicated
                # df = df.drop_duplicates(subset=['id'], keep=False)
                df = df[df['id']!='id']
                df['score'] = df['score'].str.replace('-inf', '-50000') # We need to replace "-inf" with a intenger value
                df = df[df["score"].str.contains("-50000") == False]
                # Converting string tuple to tuple
                df['score'] = df['score'].apply(ast.literal_eval)
                # split the score in the loss function and the len of pipeline
                df[['loss', 'len_pipeline']] = pd.DataFrame(df['score'].tolist(), index=df.index)
                # Sort thte dataframe given the colum of interest, the closest to 0 since it is neg log loss
                # df = df.iloc[(df['loss'] - 0).abs().argsort()]
                df['loss'] = df['loss'].abs()
                df = df.sort_values("loss")
                # Drop identical pipelines
                df = df.drop_duplicates(subset='pipeline')
                #df = df[df['loss']<=0]
                this_df = df.head(number_of_pipelines_to_consider)
                # Choose only the pipelines
                print('Solutios for this task')
                print(this_df)
                print('Pipelines for this solutions')
                print(this_df['pipeline'])
                dataset_similarity[this_task] += ['Pipeline: ' + a + ' Log loss: ' + str(b) for a, b in zip(list(this_df['pipeline']), list(this_df['loss']))]
            except Exception as e:
                print(e)

print("Number of pipelines per task")
for task in dataset_similarity:
    print('tasks: ', task)
    print('len tasks: ', len(dataset_similarity[task]))

# Save the dictionary as a JSON file
with open('data_regression.json', 'w') as f:
    json.dump(dataset_similarity, f)

