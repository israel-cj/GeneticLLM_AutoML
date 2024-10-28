import joblib
import os
import math
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.model_selection import cross_val_score
import time

# 2. Helper function to measure inference time
def measure_inference_times(infer_fn, data_batches):
    times = []
    for batch in data_batches:
        start = time.time()
        infer_fn(batch)
        times.append(time.time() - start)
    return np.mean(times)

# 3. Infer function to measure inference time for your sklearn pipeline
def infer(data: pd.DataFrame):
    return pipe.predict_proba(data)  # Adjust to your task (classification or regression)

# 4. Objective function (score and inference time)
def evaluate_individual(individual, X, y, inference_batches):
    # Set hyperparameters from the individual
    pipe.set_params(**individual)

    # Measure score (cross-validation)
    scores = cross_val_score(pipe, X, y, cv=3, scoring='accuracy')  # Adjust scoring to your needs
    mean_score = np.mean(scores)

    # Measure inference time
    inference_time = measure_inference_times(infer, inference_batches)

    return mean_score, inference_time  # First to maximize, second to minimize

def split_pipeline(pipe):
    steps = pipe.steps
    components = {}
    for step_name, step_obj in steps:
        components[step_name] = step_obj
    return components


def get_numeric_hyperparams_and_boolean(component):
    hyperparams = component.get_params()
    numeric_hyperparams = {param_name: param_value for param_name, param_value in hyperparams.items() if isinstance(param_value, (int, float))}
    return numeric_hyperparams


def get_numeric_hyperparams(component):
    hyperparams = component.get_params()
    numeric_hyperparams = {param_name: param_value for param_name, param_value in hyperparams.items() 
                           if isinstance(param_value, (int, float)) and not isinstance(param_value, bool)}
    return numeric_hyperparams

def print_hyperparams_with_types(component):
    hyperparams = component.get_params()
    for param_name, param_value in hyperparams.items():
        param_type = type(param_value).__name__
        print(f"Hyperparameter: {param_name}, Value: {param_value}, Type: {param_type}")


def flatten_and_convert_boolean(list_of_lists):
    result = []
    for sublist in list_of_lists:
        for item in sublist:
            if isinstance(item, bool):
                result.append(1 if item else 0)
            elif isinstance(item, float) and math.isnan(item):
                result.append(0.5)
            else:
                result.append(item)
    return result


# Define file paths
x_path = os.path.join(os.getcwd(), 'X.pkl')
y_path = os.path.join(os.getcwd(), 'y.pkl')
pipe_path = os.path.join(os.getcwd(), 'pipe.pkl')

# Load objects
X = joblib.load(x_path)
y = joblib.load(y_path)
pipe = joblib.load(pipe_path)

# hyperparams = pipe.get_params()
# print(hyperparams)

# Convert hyperparameters dictionary to a list of values
# hyperparams_list = list(hyperparams.values())

# Example usage
components = split_pipeline(pipe)
dicionary_components_hyperparams = {}
for name, component in components.items():
    # numeric_hyperparams = get_numeric_hyperparams(component)
    # dicionary_components_hyperparams[name] = numeric_hyperparams
    numeric_boolean_hyperparams = get_numeric_hyperparams_and_boolean(component)
    dicionary_components_hyperparams[name] = numeric_boolean_hyperparams

print("dictionary")
print(dicionary_components_hyperparams)

first_individual = []
for compo_dict in dicionary_components_hyperparams:
    first_individual.append(list(dicionary_components_hyperparams[compo_dict].values()))

print("first_individual")
print(first_individual)

output_list = flatten_and_convert_boolean(first_individual)
print("output_list")
print(output_list)