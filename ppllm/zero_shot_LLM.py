from openai import OpenAI
from sklearn.model_selection import train_test_split
from .run_llm_code import run_llm_code_preprocessing

client = OpenAI(api_key="")

def execute_and_evaluate_code_block(X, y, code, task):
    train_size = 0.25 if len(X) > 100000 else 0.75

    split_params = {
        "train_size": train_size,
        "random_state": 0
    }

    if task == "classification":
        split_params["stratify"] = y

    X_train, X_test, y_train, y_test = train_test_split(X, y, **split_params)

    try:
        pipe = run_llm_code_preprocessing(code, X_train, y_train)
        performance = pipe.score(X_test, y_test)
    except Exception as e:
        pipe = None
        return e, None, None

    return None, performance, pipe

def get_dataset_info(X, y, task, n=10):
    output = ""
    # Number of instances
    num_instances = X.shape[0]
    # Number of features
    num_features = X.shape[1]
    # Number of missing values
    num_missing = X.isnull().sum().sum()
    # First n column names
    first_n_columns = X.columns[:n]
    # Example values from the first n columns
    example_values = X[first_n_columns].head(1).values.tolist()
    # n values from the target y
    target_values = y[:n].values.tolist()

    output += f"\nNumber of instances: {num_instances}\n"
    output += f"Type of task: {task}\n"
    output += f"Number of features: {num_features}\n"
    output += f"Number of missing values: {num_missing}\n"
    output += f"First {n} column names: {first_n_columns}\n"
    output += f"Example values from the first {n} columns: {example_values}\n"
    output += f"First {n} values from the target y: {target_values}\n"

    output += f"""
    The dataframe split in ‘X_train’ and ‘y_train’ is loaded in memory.
    This code was written by an expert data scientist working to create a suitable pipeline (preprocessing techniques and estimator) for such dataframe, the task is ‘{task}’. It is a snippet of code that imports the packages necessary to create a ‘sklearn’ pipeline 
    You can only use models from the sklearn.ensemble module, e.g. Gradient Boosting Machines, Random Forest, Extra Trees, etc.
    
    Code formatting for each pipeline created:
    ```python
    (Import ‘sklearn’ packages to create a pipeline object called 'pipe'. In addition, call its respective 'fit' function to feed the model with 'X_train' and 'y_train'
    Along with the necessary packages, always call 'make_pipeline' from sklearn.
    Usually is good to include 'SimpleImputer' in the pipeline but if 'ColumnTransformer' will be use, do it after it.
    ```end

    Each codeblock generates exactly one useful pipeline. 
    Each codeblock ends with "```end" and starts with "```python"
    Codeblock:
    """

    return output

def generate_code(model, message):
    completion = client.chat.completions.create(
        # model='ft:gpt-3.5-turbo-0613:personal:amlb-metafeatures:8qMzrgHq', #finetuned model for datasets name
        model = model,
        messages=message,
        temperature=1,
        max_tokens=1000,
    )
    code = completion.choices[0].message.content
    code = code.replace("```python", "").replace("```", "").replace("end", "")
    return code

# def query(payload):
#     response = requests.post(API_URL, headers=headers, json=payload)
#     return response.json()

def generated_zero_shot_pipeline(X, y, task='classification', model='gpt-4o-mini'):
    prompt_mf = get_dataset_info(X, y, task)

    message_sklearn = [
        {"role": "system",
         "content": "You are an expert data scientist, where given the metafeatures of a dataset you will return the a single sklean tree based pipeline (from sklearn.ensemble) that will work on it. You answer only by generating code. Answer as concisely as possible."},
        {"role": "user", "content": prompt_mf}
    ]
    for counter_working_model in range(5): # Five chances
        try:
            code_skelarn = generate_code(model, message_sklearn)
        except Exception as e:
            print("Error in LLM API." + str(e))
            code_skelarn = None

        e, performance, pipe = execute_and_evaluate_code_block(X, y, code_skelarn, task)
        if e is not None:
            message_sklearn += [
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

    return pipe


if __name__ == '__main__':
    import openml

    dataset = openml.datasets.get_dataset(1111)  # 1111 = 'KDDCup09_appetency'
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    this_pipeline = generated_zero_shot_pipeline(X, y, 'classification')
    print(this_pipeline)

