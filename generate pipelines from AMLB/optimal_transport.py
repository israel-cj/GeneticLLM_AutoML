# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 11:16:44 2021

@author: 20210595
"""

import openml
import json
import pickle
import numpy as np
from ott.geometry import pointcloud
from sklearn.decomposition import FastICA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from dirty_cat import TableVectorizer
from scipy.sparse import isspmatrix

# Regression
suite = openml.study.get_suite(269)
tasks = suite.tasks
datasets = []
dict_representation = dict()
max_samples = 5000
for task_id in tasks:
    task = openml.tasks.get_task(task_id)
    print('task_id', task_id)
    if task_id == 360932:
        print('a')
    datasetID = task.dataset_id
    dataset = openml.datasets.get_dataset(datasetID)
    # Join X_train and y_train into a dataframe
    X_train, y_train, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    # Randomly sample the DataFrame to get at most "max_samples" samples

    if X_train.shape[1] > 1000:  # If the dimensionality of the dataset is to large we limited even more
        min_size_sample = min(500, len(X_train) - 1)
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=min_size_sample)

    if len(X_train) > 1000 and X_train.shape[1] > 500:  # If the dimensionality of the dataset is to large we limited even more
        min_size_sample = min(1000, len(X_train) - 1)
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=min_size_sample)

    if len(X_train) > 2000 and X_train.shape[1] > 100:  # If the dimensionality of the dataset is to large we limited even more
        min_size_sample = min(1500, len(X_train) - 1)
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=min_size_sample)

    if len(X_train) > max_samples:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=max_samples)

    # y_train = y_train.sparse.to_dense() # There is at least 1 dataset in openml with y as sparse float pandas
    table_vec = TableVectorizer(auto_cast=True)
    try:
        transformed_data = table_vec.fit_transform(X_train, y_train)

        if isspmatrix(transformed_data):
            transformed_data = transformed_data.toarray()
        imp_mean = SimpleImputer(strategy='mean')
        imp_mean.fit(transformed_data)
        transformed_data = imp_mean.transform(transformed_data)
        # The transformed_data is now a NumPy array with all preprocessing applied
        transformed_dataset = FastICA().fit_transform(transformed_data)
        geom_yy = pointcloud.PointCloud(transformed_dataset)
        # Print the joint probability
        print(f"Optimal transport value for this dataset: {geom_yy}")
        dict_representation[dataset.name] = geom_yy
    except Exception as e:
        print(e)

with open('regression_optimal_transport_light.pickle', 'wb') as handle:
    pickle.dump(dict_representation, handle, protocol=pickle.HIGHEST_PROTOCOL)


####

# Classification
suite = openml.study.get_suite(271)
tasks = suite.tasks
datasets = []
dict_representation = dict()
max_samples = 5000
for task_id in tasks:
    task = openml.tasks.get_task(task_id)
    print('task_id', task_id)
    datasetID = task.dataset_id
    dataset = openml.datasets.get_dataset(datasetID)
    # Join X_train and y_train into a dataframe
    X_train, y_train, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    # Randomly sample the DataFrame to get at most "max_samples" samples
    if X_train.shape[1] > 1000:  # If the dimensionality of the dataset is to large we limited even more
        min_size_sample = min(500, len(X_train) - 1)
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=min_size_sample)

    if len(X_train) > 1000 and X_train.shape[1] > 500:  # If the dimensionality of the dataset is to large we limited even more
        min_size_sample = min(1000, len(X_train) - 1)
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=min_size_sample)

    if len(X_train) > 2000 and X_train.shape[1] > 100:  # If the dimensionality of the dataset is to large we limited even more
        min_size_sample = min(1500, len(X_train) - 1)
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=min_size_sample)

    if len(X_train) > max_samples:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=max_samples)
    #try:
    table_vec = TableVectorizer(auto_cast=True)
    transformed_data = table_vec.fit_transform(X_train, y_train)

    if isspmatrix(transformed_data):
        transformed_data = transformed_data.toarray()
    imp_mean = SimpleImputer(strategy='mean')
    imp_mean.fit(transformed_data)
    transformed_data = imp_mean.transform(transformed_data)
    # The transformed_data is now a NumPy array with all preprocessing applied
    transformed_dataset = FastICA().fit_transform(transformed_data)
    geom_yy = pointcloud.PointCloud(transformed_dataset)

    print(f"Optimal transport value for this dataset: {geom_yy}")
    dict_representation[dataset.name] = geom_yy
    # except Exception as e:
    #     print(e)

with open('classification_optimal_transport_light.pickle', 'wb') as handle:
    pickle.dump(dict_representation, handle, protocol=pickle.HIGHEST_PROTOCOL)

