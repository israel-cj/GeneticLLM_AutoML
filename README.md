# GeneticLLM_AutoML

This project is based on utilizing LLMs and Genetic Programming (GP) to create a model for a given dataset. The LLM specialize in creating a zero-shot model and using GP for further optimization. It serves as a tool to simplify the process of creating a machine learning model with just a few lines of code, where the entire pipeline is constructed and optimized by these combined tools.

For the framework there are two main requirements: determining whether the problem is classification or regression and providing the dataset split into 'X' and 'y'. Don't forget to set up the OPENAI KEY in the ENV.


## Figures and Data

The Figures and data are in the folder "Figures and data", please run the notebook of your interest to reproduce the paper.

## Classification

```python

import openai
import openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from ppllm import PP_LLM

# Define the OPENAI KEY IN THE ENV

dataset = openml.datasets.get_dataset(40983) # 40983 is Wilt dataset: https://www.openml.org/search?type=data&status=active&id=40983
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="dataframe", target=dataset.default_target_attribute
)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

### Setup and Run LLM pipeline - This will be billed to your OpenAI Account!
automl = PP_LLM(
    llm_model="gpt-4o-mini",
    max_total_time = 300,
    )

automl.fit(X_train, y_train)

# This process is done only once
y_pred = automl.predict(X_test)
acc = accuracy_score(y_pred, y_test)
print(f'LLM Pipeline accuracy {acc}')

```

## Regression

```python
import openai
import openml
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from ppllm import PP_LLM


# Define the OPENAI KEY IN THE ENV
type_task = 'regression'
dataset = openml.datasets.get_dataset(41021) # 41021 is Moneyball dataset: https://www.openml.org/search?type=data&status=active&id=41021
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="dataframe", target=dataset.default_target_attribute
)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

### Setup and Run LLM pipeline - This will be billed to your OpenAI Account!
automl = PP_LLM(
    llm_model="gpt-4o-mini",
    task=type_task,
    max_total_time=300
    )

automl.fit(X_train, y_train)

# This process is done only once
y_pred = automl.predict(X_test)
print("LLM Pipeline MSE:", mean_squared_error(y_test, y_pred))

```
