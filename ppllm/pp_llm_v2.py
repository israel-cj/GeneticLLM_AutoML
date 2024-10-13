import datetime
import csv
import time
import random
from openai import OpenAI
from sklearn.model_selection import train_test_split
from .run_llm_code import run_llm_code_preprocessing
from .similarity_gpt import get_name
from .amlb_models import data_classification, data_regression

client = OpenAI(api_key='')

list_pipelines = []

def get_code_from_name(X, y, task):
    name = get_name(X, y, task)
    if task == 'classification':
        try:
            str_pipeline = data_classification[name][0]
        except Exception as e:
            print(e)
            str_pipeline= data_classification[list(data_classification.keys())[random.randint(0, 70)]][0]
    if task == 'regression':
        try:
            str_pipeline= data_regression[name][0]
        except Exception as e:
            print(e)
            str_pipeline = data_regression[list(data_regression.keys())[random.randint(0, 30)]][0]
    return str_pipeline


def get_prompt_sklearn(code, task):
    return f"""
Consider the next string for a {task} task:

{code}

Your task will be to build an identical pipe called 'pipe'. You respond only by providing code, e.g.
```python
(Import ‘sklearn’ packages to create a pipeline object called 'pipe'. In addition, call its respective 'fit' function to feed the model with 'X_train' and 'y_train'
Along with the necessary packages, always call 'make_pipeline' from sklearn.
```end

Each codeblock generates exactly one useful pipeline.". 
Each codeblock ends with "```end" and starts with "```python"
Codeblock:
"""

def get_prompt_ensemble(code, task):
    return f"""
The pipeline for a {task} task is:

{code}

Your task will be to build a pipeline 'pipe' with different estimator, but same preprocessing. You respond only by providing code, e.g.
```python
(Import ‘sklearn’ packages to create a pipeline object called 'pipe'. In addition, call its respective 'fit' function to feed the model with 'X_train' and 'y_train'
Along with the necessary packages, always call 'make_pipeline' from sklearn.
```end

Each codeblock generates exactly one useful pipeline.". 
Each codeblock ends with "```end" and starts with "```python".
Codeblock:
"""

def generate_pipelines(
        X,
        y,
        name=None,
        model="gpt-3.5-turbo",
        display_method="markdown",
        iterations=4,
        task="classification",
        identifier=None,
        ensemble=False,
        y_origin= None
):
    global list_pipelines # To make it available to sklearn_wrapper in case the time out is reached
    def format_for_display(code):
        code = code.replace("```python", "").replace("```", "").replace("<end>", "")
        return code

    if display_method == "markdown":
        from IPython.display import display, Markdown

        display_method = lambda x: display(Markdown(x))
    else:

        display_method = print

    def generate_code(messages):
        if model == "skip":
            return ""
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            stop=["```end"],
            temperature=0.5,
            max_tokens=1000,
        )
        code = completion.choices[0].message.content
        code = code.replace("```python", "").replace("```", "").replace("<end>", "")
        return code

    def generate_code_ensemble(messages):
        if model == "skip":
            return ""
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            stop=["```end"],
            temperature=0.9, # let's see how this works
            max_tokens=1000,
        )
        code = completion.choices[0].message.content
        code = code.replace("```python", "").replace("```", "").replace("<end>", "")
        return code

    def execute_and_evaluate_code_block(code):
        train_size = 0.75
        if len(X)>100000:
            train_size = 0.25
        if task == "classification":
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size,  stratify=y, random_state=0)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=0)
        try:
            pipe = run_llm_code_preprocessing(
                code,
                X_train,
                y_train,
            )
            performance = pipe.score(X_test, y_test)
        except Exception as e:
            pipe = None
            display_method(f"Error in code execution. {type(e)} {e}")
            display_method(f"```python\n{format_for_display(code)}\n```\n")
            return e, None, None

        return None, performance, pipe

    display_method(f"*Dataset with specific description, task:*\n {task}")
    list_codeblocks = []
    list_performance = []

    code = get_code_from_name(X, y_origin, task)

    messages_sklearn = [
        {
            "role": "system",
            "content": "You are an expert data science assistant creating a sklearn Pipeline for a dataset X_train, y_train given a string. You answer only by generating code.",
        },
        {
            "role": "user",
            "content": get_prompt_sklearn(code, task),
        },
    ]
    for counter_working_model in range(4):
        try:
            code_skelarn = generate_code(messages_sklearn)
        except Exception as e:
            display_method("Error in LLM API." + str(e))
            time.sleep(60)  # Wait 1 minute before next request

        e, performance, pipe = execute_and_evaluate_code_block(code_skelarn)
        if e is not None:
            messages_sklearn += [
                {"role": "assistant", "content": code_skelarn},
                {
                    "role": "user",
                    "content": f"""Code execution failed with error: {type(e)} {e}.\n Code: ```python{code_skelarn}```\n Generate new pipeline (fixing error?):
                                            ```python
                                            """,
                },
            ]
        if e is None:
            break

    if isinstance(performance, float):
        valid_pipeline = True
        pipeline_sentence = f"The code was executed and generated a ´pipe´ with score {performance}"
    else:
        valid_pipeline = False
        pipeline_sentence = "The last code did not generate a valid ´pipe´, it was discarded."

    display_method(
        "\n"
        + f"*Valid pipeline: {str(valid_pipeline)}*\n"
        + f"```python\n{format_for_display(code)}\n```\n"
        + f"Performance {performance} \n"
        + f"{pipeline_sentence}\n"
        + f"\n"
    )

    if e is not None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Write the data to a CSV file
        with open(f'pipelines_{identifier}.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([timestamp, name, code, e])

    if e is None:
        list_codeblocks.append(code_skelarn)
        list_performance.append(performance)
        list_pipelines.append(pipe)
        print('The performance of this pipeline is: ', performance)
        # # Get news similar_pipelines to explore more
        # similar_pipelines = TransferedPipelines(X_train=X, y_train=y, task=task, number_of_pipelines=3)
        # Get the current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Write the data to a CSV file
        with open(f'pipelines_{identifier}.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([timestamp, name, code, str(performance)])

    if ensemble == True:
        print("Lets create the ensemble")
        counter_models = 0
        messages_ensemble = [
            {
                "role": "system",
                "content": "You are an expert data science assistant who creates a pipeline with a different estimator compared to the one given. You answer only by generating code.",
            },
            {
                "role": "user",
                "content": get_prompt_ensemble(code_skelarn, task),
            },
        ]

        while counter_models<iterations:
            try:
                code_ensemble = generate_code_ensemble(messages_ensemble)
            except Exception as e:
                display_method("Error in LLM API." + str(e))
                time.sleep(60)  # Wait 1 minute before next request

            e, performance, pipe = execute_and_evaluate_code_block(code_ensemble)

            if e is not None:
                messages_ensemble += [
                    {"role": "assistant", "content": code_ensemble},
                    {
                        "role": "user",
                        "content": f"""Code execution failed with error: {type(e)} {e}.\n Code: ```python{code_ensemble}```\n Generate new pipeline (fixing error?):
                                                            ```python
                                                            """,
                    },
                ]
            if e is None:
                list_codeblocks.append(code_ensemble)
                list_performance.append(performance)
                list_pipelines.append(pipe)
                print('The performance of this pipeline is: ', performance)
                # # Get news similar_pipelines to explore more
                # similar_pipelines = TransferedPipelines(X_train=X, y_train=y, task=task, number_of_pipelines=3)
                # Get the current timestamp
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Write the data to a CSV file
                with open(f'pipelines_{identifier}.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([timestamp, name, code_ensemble, str(performance)])

            counter_models+=1

    # return list_codeblocks, list_performance,
    return list_codeblocks, list_performance

