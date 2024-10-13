import requests
from openai import OpenAI
from .openml_suites_name import data_automl

# api_token = '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'
# API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/msmarco-distilbert-base-tas-b"
# headers = {"Authorization": f"Bearer {api_token}"}
client = OpenAI(api_key='')


def get_dataset_info(X, y, y_name, n=10):
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
    output += f"Number of features: {num_features}\n"
    output += f"Number of missing values: {num_missing}\n"
    output += f"First {n} column names: {first_n_columns}\n"
    output += f"Example values from the first {n} columns: {example_values}\n"
    output += f"Target variable (y) name: {y_name}\n"
    output += f"First {n} values from the target y: {target_values}\n"

    return output

def generate_code(prompt_mf):

    completion = client.chat.completions.create(
        model='ft:gpt-3.5-turbo-0613:personal:amlb-metafeatures:8qMzrgHq',
        messages=[
            {"role": "system",
             "content": "You are an expert data scientist, where given the metafeatures of a dataset you will return the name of the dataset."},
            {"role": "user", "content": prompt_mf}
        ],
        temperature=0.2,
        max_tokens=1000,
    )
    return completion.choices[0].message.content

# def query(payload):
#     response = requests.post(API_URL, headers=headers, json=payload)
#     return response.json()

def get_name(X, y, task='classification'):
    # # Load the JSON data which contains the pipelines from the AutoMLBenchmark
    # if task == 'classification':
    #     list_name_datasets = data_automl['classification']
    # else:
    #     list_name_datasets = data_automl['regression']
    # prompt_mf = get_dataset_info(X, y[y.columns[0]], y.columns[0])
    prompt_mf = get_dataset_info(X, y, y.name)
    # data = query(
    #     {
    #         "inputs": {
    #             "source_sentence": prompt_mf,
    #             "sentences": list_name_datasets
    #         }
    #     }
    # )
    #
    # highest_index_of_data = data.index(max(data))
    # name_dataset = list_name_datasets[highest_index_of_data]
    try:
        name_dataset = generate_code(prompt_mf)
    except Exception as e:
        print("Error in LLM API." + str(e))
        name_dataset = None

    return name_dataset


if __name__ == '__main__':
    import openml

    dataset = openml.datasets.get_dataset(1111)  # 1111 = 'KDDCup09_appetency'
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    this_name = get_name(X, y, 'classification')
    print(this_name)

